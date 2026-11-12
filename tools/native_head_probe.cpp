// Standalone original head-only experiment, not an inference-engine backend.
// Pool and FP16 dot accumulation are adapted from this repository's engine/core.cpp
// SHA25686f84760e759e7a3b0b9e3b6115118b5e85c5bb1da42f4d9c37c160358067281.
// No llama/ggml kernels. Bounds require independent review before execution.
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
constexpr int V=151936,H=1024;
using Half=__fp16;
using Clock=std::chrono::steady_clock;
static_assert(sizeof(Half)==2 && sizeof(float)==4 && sizeof(double)==8);
static_assert(std::endian::native==std::endian::little);
void require(bool condition,const std::string& message) {if(!condition)throw std::runtime_error(message);}
uint64_t fpcr() {uint64_t value;__asm__ volatile("mrs %0, fpcr":"=r"(value));return value;}
// Require the entire register to be zero. No unreviewed newer ARM alternative
// FP behavior is silently accepted; nonzero states take the dense fallback.
bool supported_fpcr(uint64_t value){return value==0;}
bool has_dotprod(){int value=0;size_t size=sizeof(value);return sysctlbyname("hw.optional.arm.FEAT_DotProd",&value,&size,nullptr,0)==0&&value==1;}

class Pool {
    struct alignas(128) Completion {std::atomic<uint64_t> generation{0};};
    int n_;std::vector<std::thread> threads_;std::mutex mutex_;std::condition_variable start_;
    std::atomic<bool> stop_{false};alignas(128) std::atomic<uint64_t> generation_{0};
    std::array<Completion,64> completed_{};int parked_=0;size_t count_=0;
    void* context_=nullptr;void(*job_)(void*,size_t,size_t,int)=nullptr;std::exception_ptr error_;
    void work(int i){uint64_t seen=0;for(;;){
        const auto deadline=Clock::now()+std::chrono::microseconds(50);unsigned spins=0;
        while(generation_.load(std::memory_order_acquire)==seen&&!stop_.load(std::memory_order_relaxed)){
            __asm__ volatile("yield":::"memory");if((++spins&63)==0&&Clock::now()>=deadline)break;}
        if(stop_.load(std::memory_order_acquire))return;
        if(generation_.load(std::memory_order_acquire)==seen){std::unique_lock lock(mutex_);++parked_;
            start_.wait(lock,[&]{return stop_.load(std::memory_order_relaxed)||generation_.load(std::memory_order_acquire)!=seen;});
            --parked_;if(stop_.load(std::memory_order_relaxed))return;}
        const uint64_t next=generation_.load(std::memory_order_acquire);const size_t count=count_;
        try{job_(context_,count*i/n_,count*(i+1)/n_,i);}catch(...){std::lock_guard lock(mutex_);if(!error_)error_=std::current_exception();}
        seen=next;completed_[i].generation.store(seen,std::memory_order_release);
    }}
public:
    explicit Pool(int n):n_(n){try{for(int i=1;i<n_;++i)threads_.emplace_back([this,i]{work(i);});}catch(...){shutdown();throw;}}
    ~Pool(){shutdown();}
    void shutdown(){{std::lock_guard lock(mutex_);stop_.store(true,std::memory_order_release);}start_.notify_all();for(auto& t:threads_)if(t.joinable())t.join();}
    template<class Fn>void run(size_t count,Fn fn){uint64_t next;bool wake;{
        std::lock_guard lock(mutex_);context_=&fn;job_=[](void* p,size_t b,size_t e,int i){(*static_cast<Fn*>(p))(b,e,i);};
        count_=count;error_=nullptr;next=generation_.load(std::memory_order_relaxed)+1;
        generation_.store(next,std::memory_order_release);wake=parked_!=0;}
        if(wake)start_.notify_all();try{job_(context_,0,count/n_,0);}catch(...){std::lock_guard lock(mutex_);if(!error_)error_=std::current_exception();}
        for(int i=1;i<n_;++i){unsigned spins=0;while(completed_[i].generation.load(std::memory_order_acquire)!=next){
            __asm__ volatile("yield":::"memory");if((++spins&1023)==0)std::this_thread::yield();}}
        job_=nullptr;context_=nullptr;if(error_)std::rethrow_exception(error_);
    }
};

struct Mapping {
    void* data=MAP_FAILED;size_t length=0;
    explicit Mapping(const std::string& path){int fd=open(path.c_str(),O_RDONLY);require(fd>=0,"Cannot open pack");
        struct stat st{};if(fstat(fd,&st)!=0||st.st_size<64){close(fd);throw std::runtime_error("Invalid pack size");}
        length=static_cast<size_t>(st.st_size);data=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);close(fd);require(data!=MAP_FAILED,"Cannot map pack");}
    ~Mapping(){if(data!=MAP_FAILED)munmap(data,length);}
};
template<class T>std::vector<T> read_array(const std::string& path,size_t count){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);require(bool(stream)&&stream.tellg()==std::streamoff(count*sizeof(T)),"Wrong array size: "+path);
    stream.seekg(0);std::vector<T> values(count);stream.read(reinterpret_cast<char*>(values.data()),count*sizeof(T));require(bool(stream),"Array read failed");return values;
}
float dot(const float* x,const Half* w){float32x4_t a=vdupq_n_f32(0),b=a;
    for(int j=0;j<H;j+=8){const float16x8_t v=vld1q_f16(w+j);
        a=vfmaq_f32(a,vld1q_f32(x+j),vcvt_f32_f16(vget_low_f16(v)));
        b=vfmaq_f32(b,vld1q_f32(x+j+4),vcvt_f32_f16(vget_high_f16(v)));}
    return vaddvq_f32(vaddq_f32(a,b));
}
// Generalized row addresses; per-row FMA/reduction order is unchanged.
void dot4(const float* x,const std::array<const Half*,4>& w,float* y){
    float32x4_t a0=vdupq_n_f32(0),b0=a0,a1=a0,b1=a0,a2=a0,b2=a0,a3=a0,b3=a0;
    for(int j=0;j<H;j+=8){const auto xl=vld1q_f32(x+j),xh=vld1q_f32(x+j+4);
        const auto w0=vld1q_f16(w[0]+j),w1=vld1q_f16(w[1]+j),w2=vld1q_f16(w[2]+j),w3=vld1q_f16(w[3]+j);
        a0=vfmaq_f32(a0,xl,vcvt_f32_f16(vget_low_f16(w0)));b0=vfmaq_f32(b0,xh,vcvt_f32_f16(vget_high_f16(w0)));
        a1=vfmaq_f32(a1,xl,vcvt_f32_f16(vget_low_f16(w1)));b1=vfmaq_f32(b1,xh,vcvt_f32_f16(vget_high_f16(w1)));
        a2=vfmaq_f32(a2,xl,vcvt_f32_f16(vget_low_f16(w2)));b2=vfmaq_f32(b2,xh,vcvt_f32_f16(vget_high_f16(w2)));
        a3=vfmaq_f32(a3,xl,vcvt_f32_f16(vget_low_f16(w3)));b3=vfmaq_f32(b3,xh,vcvt_f32_f16(vget_high_f16(w3)));}
    y[0]=vaddvq_f32(vaddq_f32(a0,b0));y[1]=vaddvq_f32(vaddq_f32(a1,b1));y[2]=vaddvq_f32(vaddq_f32(a2,b2));y[3]=vaddvq_f32(vaddq_f32(a3,b3));
}
// Finite positive FP64 inputs here stay below1e50 by F16/F32 input bounds.
// Bit increments implement nextafter(+infinity), including positive zero.
double up_positive(double value){return std::bit_cast<double>(std::bit_cast<uint64_t>(value)+1);}
double up_signed(double value){if(value==0)return std::numeric_limits<double>::denorm_min();
    uint64_t bits=std::bit_cast<uint64_t>(value);bits+=value>0?uint64_t(1):uint64_t(-1);return std::bit_cast<double>(bits);}
float float_up(float value){if(value==std::numeric_limits<float>::infinity())return value;if(value==0)return std::numeric_limits<float>::denorm_min();
    uint32_t bits=std::bit_cast<uint32_t>(value);bits+=value>0?uint32_t(1):uint32_t(-1);return std::bit_cast<float>(bits);}
float enclose_float_up(double value){
    const double maximum=double(std::numeric_limits<float>::max());
    if(value>maximum)return std::numeric_limits<float>::infinity();
    if(value< -maximum)return -std::numeric_limits<float>::max();
    float result=static_cast<float>(value);if(double(result)<value)result=float_up(result);return result;
}
double power_scale(double maximum){if(maximum==0)return 1;int exponent;const double fraction=std::frexp(maximum/127.0,&exponent);return std::ldexp(1.0,exponent-(fraction==.5));}
int8_t quantize(double value,double scale){return static_cast<int8_t>(std::clamp(std::nearbyint(value/scale),-127.0,127.0));}
float score(float value,int id,bool chat,const std::vector<uint8_t>& seen){return chat?(value-(seen[id]?1.5f:0.f))/.7f:value;}
struct Item {float value,logit;int id;};
struct Scored {float value;int id;};
bool better(const Scored& a,const Scored& b){return a.value>b.value||(a.value==b.value&&a.id<b.id);}
struct Approx {double value;int id;};
bool approximate_better(const Approx& a,const Approx& b){return a.value>b.value||(a.value==b.value&&a.id<b.id);}
struct Meta {float scale,residual,l1;};static_assert(sizeof(Meta)==12);
struct Result {std::vector<Item> top;size_t rescored=V,unsafe=0;bool fallback=false;};
bool equal(const Result& a,const Result& b){if(a.top.size()!=b.top.size())return false;for(size_t i=0;i<a.top.size();++i)
    if(a.top[i].id!=b.top[i].id||std::bit_cast<uint32_t>(a.top[i].logit)!=std::bit_cast<uint32_t>(b.top[i].logit)
       ||std::bit_cast<uint32_t>(a.top[i].value)!=std::bit_cast<uint32_t>(b.top[i].value))return false;return true;}

class Probe {
    Mapping mapping_;const Half* weights_=nullptr;Pool pool_;int threads_;
    std::vector<float> hidden_,saved_,dense_,exact_;
    std::vector<uint8_t> seen_,marked_,unsafe_;
    std::vector<int8_t> cache_;std::vector<Meta> metadata_;
    std::vector<Approx> approximate_;std::vector<Scored> scores_;
    std::vector<int> candidates_;std::vector<double> upper_;
    bool enabled_=false;
    void rescore(size_t begin,size_t end){size_t i=begin;for(;i+4<=end;i+=4){float y[4];
        dot4(hidden_.data(),{weights_+size_t(candidates_[i])*H,weights_+size_t(candidates_[i+1])*H,
            weights_+size_t(candidates_[i+2])*H,weights_+size_t(candidates_[i+3])*H},y);
        for(int j=0;j<4;++j)exact_[candidates_[i+j]]=y[j];}
        for(;i<end;++i)exact_[candidates_[i]]=dot(hidden_.data(),weights_+size_t(candidates_[i])*H);
    }
    Result select(bool chat,const std::vector<float>& values,const std::vector<int>* subset=nullptr){
        const size_t count=subset?subset->size():V;const int k=chat?20:1;
        if(!chat){Item best{-std::numeric_limits<float>::infinity(),0,V};
            for(size_t i=0;i<count;++i){const int id=subset?(*subset)[i]:int(i);const float value=values[id];
                if(!std::isfinite(value))throw std::runtime_error("Nonfinite canonical logit");
                if(value>best.value||(value==best.value&&id<best.id))best={value,value,id};}
            Result result;result.top.push_back(best);return result;}
        for(size_t i=0;i<count;++i){const int id=subset?(*subset)[i]:int(i);const float value=score(values[id],id,chat,seen_);
            if(!std::isfinite(values[id])||!std::isfinite(value))throw std::runtime_error("Nonfinite canonical logit/sampler score");
            scores_[i]={value,id};}
        std::partial_sort(scores_.begin(),scores_.begin()+k,scores_.begin()+count,better);
        Result result;result.top.reserve(k);for(int i=0;i<k;++i)result.top.push_back({scores_[i].value,values[scores_[i].id],scores_[i].id});return result;
    }
public:
    std::vector<uint64_t> worker_fpcr;bool dotprod=false;double cache_build_seconds=0;
    Probe(const std::string& pack,const std::string& hidden,const std::string& logits,const std::string& seen,int threads)
        :mapping_(pack),pool_(threads),threads_(threads),hidden_(read_array<float>(hidden,H)),saved_(read_array<float>(logits,V)),
         dense_(V),exact_(V),seen_(seen.empty()?std::vector<uint8_t>(V,0):read_array<uint8_t>(seen,V)),marked_(V),unsafe_(V),
         approximate_(V),scores_(V),upper_(V),worker_fpcr(threads){
        for(float value:hidden_)require(std::isfinite(value),"Nonfinite hidden state");for(float value:saved_)require(std::isfinite(value),"Nonfinite saved logits");
        for(uint8_t value:seen_)require(value<=1,"Invalid seen mask");
        std::array<unsigned char,64> magic{};std::memcpy(magic.data(),"QWCPU01",7);require(std::memcmp(mapping_.data,magic.data(),64)==0,"Invalid pack header");
        std::ifstream index(pack+".index");std::string signature;size_t count;require(bool(index>>signature>>count)&&signature=="QWCPU01"&&count==310,"Invalid pack index");
        for(size_t i=0;i<count;++i){std::string name,type;uint64_t rows,cols,offset,length;require(bool(index>>name>>type>>rows>>cols>>offset>>length),"Truncated index");
            if(name=="model.embed_tokens.weight"){require(weights_==nullptr&&type=="F16"&&rows==V&&cols==H&&length==uint64_t(V)*H*2
                &&offset>=64&&offset%64==0&&offset<=mapping_.length&&length<=mapping_.length-offset,"Invalid head matrix");
                weights_=reinterpret_cast<const Half*>(static_cast<const unsigned char*>(mapping_.data)+offset);}}
        require(weights_!=nullptr,"Missing head matrix");candidates_.reserve(V);
        pool_.run(threads,[&](size_t,size_t,int worker){worker_fpcr[worker]=fpcr();});
        dotprod=has_dotprod();enabled_=dotprod&&supported_fpcr(fpcr())&&std::all_of(worker_fpcr.begin(),worker_fpcr.end(),supported_fpcr);
        if(!enabled_)return;
        const auto start=Clock::now();cache_.resize(size_t(V)*H);metadata_.resize(V);
        pool_.run(V,[&](size_t begin,size_t end,int){for(size_t row=begin;row<end;++row){
            double maximum=0;for(int j=0;j<H;++j){const double value=float(weights_[row*H+j]);if(!std::isfinite(value))throw std::runtime_error("Nonfinite weight");maximum=std::max(maximum,std::abs(value));}
            const double scale=power_scale(maximum);double error=0;int absolute_sum=0;
            for(int j=0;j<H;++j){const double value=float(weights_[row*H+j]);const int8_t q=quantize(value,scale);cache_[row*H+j]=q;
                absolute_sum+=std::abs(int(q));error=std::max(error,up_positive(std::abs(value-double(q)*scale)));}
            const double l1=double(absolute_sum)*scale;metadata_[row]={float(scale),enclose_float_up(error),float(l1)};
            require(double(metadata_[row].scale)==scale&&double(metadata_[row].l1)==l1,"Inexact cache metadata");}});
        cache_build_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    }
    bool routing_enabled()const{return enabled_;}
    size_t auxiliary_bytes()const{return cache_.size()+metadata_.size()*sizeof(Meta);}
    size_t seen_count()const{return std::count(seen_.begin(),seen_.end(),uint8_t(1));}
    bool matches_saved()const{return std::memcmp(dense_.data(),saved_.data(),V*sizeof(float))==0;}
    bool upper_covers_dense()const{for(int i=0;i<V;++i)if(!(double(dense_[i])<=upper_[i]))return false;return true;}
    Result dense(bool chat){pool_.run(V,[&](size_t begin,size_t end,int){size_t row=begin;for(;row+4<=end;row+=4)
        dot4(hidden_.data(),{weights_+row*H,weights_+(row+1)*H,weights_+(row+2)*H,weights_+(row+3)*H},dense_.data()+row);
        for(;row<end;++row)dense_[row]=dot(hidden_.data(),weights_+row*H);});return select(chat,dense_);}
    Result routed(bool chat){
        if(!enabled_||!supported_fpcr(fpcr())){auto result=dense(chat);result.fallback=true;return result;}
        std::array<int8_t,H> qx{};double max_x=0,sum_x=0,error_x=0;
        for(float value:hidden_){max_x=std::max(max_x,std::abs(double(value)));sum_x=up_positive(sum_x+std::abs(double(value)));}
        const double scale_x=power_scale(max_x);
        for(int j=0;j<H;++j){qx[j]=quantize(double(hidden_[j]),scale_x);error_x=std::max(error_x,up_positive(std::abs(double(hidden_[j])-double(qx[j])*scale_x)));}
        const double u=0x1p-24,gamma=up_positive((131*u)/(1-131*u)),amplification=up_positive(1+gamma);
        const double tiny=double(std::numeric_limits<float>::min()),underflow_base=up_positive(3093.0*tiny);
        std::atomic<bool> unsupported{false};
        pool_.run(V,[&](size_t begin,size_t end,int){if(!supported_fpcr(fpcr())){unsupported.store(true,std::memory_order_relaxed);return;}
            for(size_t row=begin;row<end;++row){int32x4_t accum=vdupq_n_s32(0);const int8_t* weights=cache_.data()+row*H;
                for(int j=0;j<H;j+=16)accum=vdotq_s32(accum,vld1q_s8(weights+j),vld1q_s8(qx.data()+j));
                const int32_t integer=vaddvq_s32(accum);const Meta m=metadata_[row];
                // |integer|<=1024*127^2<2^24; both scales are powers of two.
                const double center=double(integer)*(double(m.scale)*scale_x);
                // wx-ab=(w-a)x+a(x-b), hence Ew*||x||1+Ex*||a||1.
                // S=(||a||1+H*Ew)*max|x| encloses sum|w*x|. Canonical
                // eight-lane FMA/reduction has longest rounding path131.
                // Every positive bound operation rounds toward +infinity;
                // scalar contraction must stay disabled in the build.
                const double quant=up_positive(up_positive(double(m.residual)*sum_x)+up_positive(error_x*double(m.l1)));
                const double sum_w=up_positive(double(m.l1)+up_positive(double(H)*double(m.residual)));
                const double absolute_products=up_positive(sum_w*max_x);
                const double underflow=up_positive(up_positive(up_positive(tiny*sum_w)+underflow_base)*amplification);
                const double reference=up_positive(up_positive(gamma*absolute_products)+underflow);
                const bool unsafe=up_positive(absolute_products+reference)>double(std::numeric_limits<float>::max());
                unsafe_[row]=unsafe;const double total=up_positive(quant+reference);
                upper_[row]=unsafe?std::numeric_limits<double>::infinity():up_signed(up_signed(center)+total);
                approximate_[row]={chat?(center-(seen_[row]?1.5:0.0))/double(.7f):center,int(row)};
            }});
        if(unsupported.load(std::memory_order_relaxed)){auto result=dense(chat);result.fallback=true;return result;}
        const int k=chat?20:1;
        std::partial_sort(approximate_.begin(),approximate_.begin()+k,approximate_.end(),approximate_better);
        candidates_.clear();std::fill(marked_.begin(),marked_.end(),0);
        for(int i=0;i<k;++i){const int id=approximate_[i].id;candidates_.push_back(id);marked_[id]=1;}
        // Threshold is obtained ONLY from exact canonical rescoring of the seed.
        // Dense saved logits and dense_ are never read anywhere in this path.
        rescore(0,k);auto seed=select(chat,exact_,&candidates_);const float threshold=seed.top.back().value;
        for(int id=0;id<V;++id)if(!marked_[id]){
            const float upper_score=score(enclose_float_up(upper_[id]),id,chat,seen_);
            // Keep equality irrespective of token ID: conservative tie handling.
            if(upper_score>=threshold)candidates_.push_back(id);
        }
        pool_.run(candidates_.size()-k,[&](size_t begin,size_t end,int){
            if(!supported_fpcr(fpcr())){unsupported.store(true,std::memory_order_relaxed);return;}rescore(begin+k,end+k);});
        if(unsupported.load(std::memory_order_relaxed)){auto result=dense(chat);result.fallback=true;return result;}
        auto result=select(chat,exact_,&candidates_);result.rescored=candidates_.size();
        result.unsafe=std::count(unsafe_.begin(),unsafe_.end(),uint8_t(1));return result;
    }
};

struct Measurement {int round,order;bool chat,routed;double seconds;Result result;};
void write_result(std::ostream& out,const Result& result){out<<"{\"rescored_rows\":"<<result.rescored<<",\"unsafe_bound_rows\":"<<result.unsafe
    <<",\"dense_fallback\":"<<(result.fallback?"true":"false")<<",\"top\":[";for(size_t i=0;i<result.top.size();++i){if(i)out<<',';
        const auto item=result.top[i];out<<"{\"id\":"<<item.id<<",\"logit\":"<<item.logit<<",\"score\":"<<item.value<<'}';}out<<"]}";}
void save(const std::string& path,const Probe& probe,const std::vector<Measurement>& rows,int rounds,int warmups,
          bool preflight,bool completed,const std::array<Result,2>& references,const std::array<Result,2>& routes){
    const std::string temporary=path+".pending";std::ofstream out(temporary);require(bool(out),"Cannot write report");out<<std::setprecision(17);
    out<<"{\"completed\":"<<(completed?"true":"false")<<",\"preflight_passed\":"<<(preflight?"true":"false")
       <<",\"head_only\":true,\"gpu\":false,\"pytorch\":false,\"rounds\":"<<rounds<<",\"warmups\":"<<warmups
       <<",\"routing_enabled\":"<<(probe.routing_enabled()?"true":"false")<<",\"dotprod_available\":"<<(probe.dotprod?"true":"false")
       <<",\"required_fpcr\":0,\"worker_fpcr\":[";
    for(size_t i=0;i<probe.worker_fpcr.size();++i){if(i)out<<',';out<<probe.worker_fpcr[i];}
    out<<"],\"worker_fpcr_hex\":[";for(size_t i=0;i<probe.worker_fpcr.size();++i){if(i)out<<',';out<<"\"0x"<<std::hex<<probe.worker_fpcr[i]<<std::dec<<'"';}
    out<<"],\"canonical_head_bytes\":"<<uint64_t(V)*H*2<<",\"auxiliary_bytes\":"<<probe.auxiliary_bytes()<<",\"cache_build_seconds\":"<<probe.cache_build_seconds
       <<",\"seen_tokens\":"<<probe.seen_count()<<",\"correctness\":[";
    if(preflight)for(int mode=0;mode<2;++mode){if(mode)out<<',';out<<"{\"mode\":\""<<(mode?"top20":"greedy")
        <<"\",\"saved_dense_bit_identical\":true,\"runtime_top_ids_and_logits_bit_identical\":true,\"upper_coverage_checked\":"
        <<(!routes[mode].fallback?"true":"false")<<",\"dense\":";write_result(out,references[mode]);out<<",\"routed\":";write_result(out,routes[mode]);out<<'}';}
    out<<"],\"measurements\":[";
    for(size_t i=0;i<rows.size();++i){if(i)out<<',';const auto& row=rows[i];out<<"{\"round\":"<<row.round<<",\"order\":"<<row.order
        <<",\"mode\":\""<<(row.chat?"top20":"greedy")<<"\",\"configuration\":\""<<(row.routed?"routed":"dense")
        <<"\",\"seconds\":"<<row.seconds<<",\"result\":";write_result(out,row.result);out<<'}';}
    out<<"]}\n";out.close();require(bool(out)&&std::rename(temporary.c_str(),path.c_str())==0,"Cannot publish report");
}
}

int main(int argc,char** argv){try{
    std::string weights,hidden,logits,seen,output;int threads=12,rounds=10,warmups=2,validation_only=0;
    for(int i=1;i<argc;i+=2){require(i+1<argc,"Missing argument");const std::string key=argv[i],value=argv[i+1];
        if(key=="--weights")weights=value;else if(key=="--hidden")hidden=value;else if(key=="--logits")logits=value;
        else if(key=="--seen-mask")seen=value;else if(key=="--output")output=value;else if(key=="--threads")threads=std::stoi(value);
        else if(key=="--rounds")rounds=std::stoi(value);else if(key=="--warmups")warmups=std::stoi(value);
        else if(key=="--validation-only")validation_only=std::stoi(value);else throw std::runtime_error("Unknown option");}
    require(!weights.empty()&&!hidden.empty()&&!logits.empty()&&!output.empty(),"weights/hidden/logits/output required");
    require(threads>=1&&threads<=64&&rounds>=10&&rounds<=100&&warmups>=1&&warmups<=10,"Invalid thread/round/warmup count");
    require(validation_only==0||validation_only==1,"validation-only must be0 or1");
    require(access(output.c_str(),F_OK)!=0,"Preserve existing report");
    Probe probe(weights,hidden,logits,seen,threads);std::array<Result,2> references,routes;std::vector<Measurement> measurements;
    save(output,probe,measurements,rounds,warmups,false,false,references,routes);
    for(int mode=0;mode<2;++mode){references[mode]=probe.dense(mode);require(probe.matches_saved(),"Dense head differs from saved canonical logits");
        routes[mode]=probe.routed(mode);require(equal(references[mode],routes[mode]),"Routed top IDs/exact logits differ from dense");
        if(!routes[mode].fallback)require(probe.upper_covers_dense(),"An upper bound excludes its canonical dense logit");}
    save(output,probe,measurements,rounds,warmups,true,false,references,routes);
    if(validation_only){save(output,probe,measurements,rounds,warmups,true,true,references,routes);return 0;}
    for(int mode=0;mode<2;++mode){for(int warm=0;warm<warmups;++warm){
        require(equal(probe.dense(mode),references[mode])&&equal(probe.routed(mode),references[mode]),"Warmup output changed");}
        for(int round=0;round<rounds;++round){for(int order=0;order<2;++order){const bool routed=(order+round)%2!=0;
            const auto start=Clock::now();auto result=routed?probe.routed(mode):probe.dense(mode);
            const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
            require(equal(result,references[mode]),"Scored top IDs/exact logits changed");
            measurements.push_back({round,order,bool(mode),routed,seconds,std::move(result)});}
            save(output,probe,measurements,rounds,warmups,true,false,references,routes);}}
    save(output,probe,measurements,rounds,warmups,true,true,references,routes);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"native_head_probe: %s\n",error.what());return 1;}}
