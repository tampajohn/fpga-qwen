// Original CPU implementation. Matrix kernels are independent of llama/ggml.
#include "core.hpp"
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace qwen_cpu {
namespace {
constexpr int H=1024, D=128, Q=16, K=8, F=3072, L=28;
using Half=__fp16;
static_assert(sizeof(Half)==2 && sizeof(float)==4);
static_assert(std::endian::native==std::endian::little,"Weight pack requires little endian");
void require(bool yes,const std::string& message) { if(!yes) throw std::runtime_error(message); }

// Sleeping persistent workers; the submitting thread is worker zero. Each job
// completes before buffers are reused, including exceptional worker completion.
class Pool {
    int n_; std::vector<std::thread> threads_; std::mutex mutex_;
    std::condition_variable start_,done_; bool stop_=false;
    uint64_t generation_=0; int remaining_=0; size_t count_=0;
    std::function<void(size_t,size_t)> job_; std::exception_ptr error_;
public:
    explicit Pool(int n):n_(n) {
        try { for(int i=1;i<n_;++i) threads_.emplace_back([this,i] {
            uint64_t seen=0;
            for(;;) {
                std::unique_lock lock(mutex_);
                start_.wait(lock,[&]{return stop_ || generation_!=seen;});
                if(stop_) return;
                seen=generation_; auto count=count_; lock.unlock();
                try { job_(count*i/n_,count*(i+1)/n_); }
                catch(...) { std::lock_guard guard(mutex_); if(!error_) error_=std::current_exception(); }
                lock.lock(); if(--remaining_==0) done_.notify_one();
            }
        }); } catch(...) { shutdown(); throw; }
    }
    ~Pool(){shutdown();}
    void shutdown() {
        {std::lock_guard lock(mutex_);stop_=true;}
        start_.notify_all(); for(auto& t:threads_) if(t.joinable()) t.join();
    }
    template<class Fn> void run(size_t count,Fn fn) {
        {std::lock_guard lock(mutex_); job_=fn;count_=count;remaining_=n_-1;error_=nullptr;++generation_;}
        start_.notify_all();
        try {job_(0,count/n_);} catch(...) {std::lock_guard lock(mutex_);if(!error_)error_=std::current_exception();}
        std::unique_lock lock(mutex_);done_.wait(lock,[&]{return remaining_==0;});
        job_={};if(error_)std::rethrow_exception(error_);
    }
};
struct Mapping {
    void* data=MAP_FAILED;size_t length=0;
    explicit Mapping(const std::string& path) {
        int fd=open(path.c_str(),O_RDONLY);require(fd>=0,"Cannot open weights");
        struct stat st{};if(fstat(fd,&st)!=0 || st.st_size<64){close(fd);throw std::runtime_error("Invalid weights size");}
        length=static_cast<size_t>(st.st_size);data=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
        require(data!=MAP_FAILED,"Cannot mmap weights");
    }
    ~Mapping(){if(data!=MAP_FAILED)munmap(data,length);}
};
struct Tensor {int rows=0,cols=0;const Half* half=nullptr;const float* norm=nullptr;std::vector<float> wide;};
float dot(const float* x,const Half* w,int n) {
    float32x4_t a=vdupq_n_f32(0),b=a;
    for(int j=0;j<n;j+=8) {
        const float16x8_t wh=vld1q_f16(w+j);
        a=vfmaq_f32(a,vld1q_f32(x+j),vcvt_f32_f16(vget_low_f16(wh)));
        b=vfmaq_f32(b,vld1q_f32(x+j+4),vcvt_f32_f16(vget_high_f16(wh)));
    }
    return vaddvq_f32(vaddq_f32(a,b));
}
void norm(const float* x,const float* w,float* out,int n) {
    float sum=0;for(int j=0;j<n;++j)sum+=x[j]*x[j];
    const float inv=1.0f/std::sqrt(sum/n+1e-6f);
    for(int j=0;j<n;++j)out[j]=(x[j]*inv)*w[j];
}
void rope(float* x,const float* cs,const float* sn) {
    for(int j=0;j<64;++j){float a=x[j],b=x[j+64];x[j]=a*cs[j]-b*sn[j];x[j+64]=b*cs[j]+a*sn[j];}
}
}
struct Model::Impl {
    Options options;Mapping mapping;std::map<std::string,Tensor> tensors;
    std::unique_ptr<Pool> decode_pool,prefill_pool;
    std::vector<Half> keys,values;std::vector<float> cosines,sines,output;
    int pos=0;
    Impl(const std::string& path,Options opt):options(opt),mapping(path) {
        require(opt.decode_threads>=1&&opt.decode_threads<=64&&opt.prefill_threads>=1&&opt.prefill_threads<=64,"Threads must be in 1..64");
        require(opt.context>=1&&opt.context<=4096,"Context must be in 1..4096");
        std::array<unsigned char,64> magic{};std::memcpy(magic.data(),"QWCPU01",7);
        require(std::memcmp(mapping.data,magic.data(),64)==0,"Invalid weight pack header");
        std::map<std::string,std::pair<int,int>> expected;
        expected["model.embed_tokens.weight"]={vocabulary,H};expected["model.norm.weight"]={H,1};
        for(int l=0;l<L;++l){std::string p="model.layers."+std::to_string(l)+".";
            for(const char* n:{"input_layernorm","post_attention_layernorm"})expected[p+n+".weight"]={H,1};
            for(const char* n:{"q_norm","k_norm"})expected[p+"self_attn."+n+".weight"]={D,1};
            expected[p+"self_attn.q_proj.weight"]={Q*D,H};
            expected[p+"self_attn.k_proj.weight"]={K*D,H};expected[p+"self_attn.v_proj.weight"]={K*D,H};
            expected[p+"self_attn.o_proj.weight"]={H,Q*D};
            expected[p+"mlp.gate_proj.weight"]={F,H};expected[p+"mlp.up_proj.weight"]={F,H};expected[p+"mlp.down_proj.weight"]={H,F};
        }
        std::ifstream index(path+".index");std::string signature;size_t count=0;
        require(bool(index>>signature>>count)&&signature=="QWCPU01"&&count==expected.size(),"Invalid weight index header/count");
        std::vector<std::pair<size_t,size_t>> ranges;
        for(size_t i=0;i<count;++i) {
            std::string name,type;uint64_t rows,cols,offset,length;
            require(bool(index>>name>>type>>rows>>cols>>offset>>length),"Truncated weight index");
            auto it=expected.find(name);require(it!=expected.end()&&!tensors.count(name),"Unexpected or duplicate tensor: "+name);
            require(rows==static_cast<uint64_t>(it->second.first)&&cols==static_cast<uint64_t>(it->second.second),"Wrong tensor shape: "+name);
            bool matrix=cols!=1;require(type==(matrix?"F16":"F32"),"Wrong tensor dtype: "+name);
            require(length==rows*cols*(matrix?2:4)&&offset>=64&&offset%64==0&&offset<=mapping.length&&length<=mapping.length-offset,"Invalid tensor range: "+name);
            ranges.emplace_back(offset,offset+length);Tensor t;t.rows=rows;t.cols=cols;
            const auto* bytes=static_cast<const unsigned char*>(mapping.data)+offset;
            if(matrix) {
                t.half=reinterpret_cast<const Half*>(bytes);
                // Layer-only FP32 cache costs ~1.76 GB, plus ~1.2 GB mapped F16
                // weights. Output/embedding stays F16; cache avoids repeated
                // conversion/temporary worker creation in batched prefill.
                if(name!="model.embed_tokens.weight")t.wide.resize(rows*cols);
                for(size_t j=0;j<rows*cols;++j){float v=t.half[j];if(!std::isfinite(v))throw std::runtime_error("Nonfinite matrix: "+name);if(!t.wide.empty())t.wide[j]=v;}
            } else {
                t.norm=reinterpret_cast<const float*>(bytes);
                for(size_t j=0;j<rows;++j)require(std::isfinite(t.norm[j]),"Nonfinite norm: "+name);
            }
            tensors.emplace(name,std::move(t));
        }
        std::string extra;require(!(index>>extra),"Trailing weight index data");
        std::sort(ranges.begin(),ranges.end());for(size_t i=1;i<ranges.size();++i)require(ranges[i].first>=ranges[i-1].second,"Overlapping tensors");
        keys.resize(size_t(L)*opt.context*K*D);values.resize(keys.size());
        cosines.resize(size_t(opt.context)*64);sines.resize(cosines.size());
        for(int p=0;p<opt.context;++p)for(int j=0;j<64;++j){double angle=p/std::pow(1000000.0,2.0*j/D);cosines[p*64+j]=std::cos(angle);sines[p*64+j]=std::sin(angle);}
        decode_pool=std::make_unique<Pool>(opt.decode_threads);prefill_pool=std::make_unique<Pool>(opt.prefill_threads);
    }
    Tensor& tensor(const std::string& name){return tensors.at(name);}
    void project(Tensor& w,const float* x,float* y,int batch,Pool& pool) {
        if(batch>1&&!w.wide.empty()) {
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,batch,w.rows,w.cols,1.0f,x,w.cols,w.wide.data(),w.cols,0.0f,y,w.rows);
        } else {
            pool.run(size_t(batch)*w.rows,[&](size_t begin,size_t end){for(size_t i=begin;i<end;++i)y[i]=dot(x+(i/w.rows)*w.cols,w.half+(i%w.rows)*w.cols,w.cols);});
        }
    }
    size_t cache_offset(int layer,int position,int head=0)const {return ((size_t(layer)*options.context+position)*K+head)*D;}
    void chunk(const int* ids,int batch,Pool& pool) {
        std::vector<float> x(size_t(batch)*H),h(x.size()),q(size_t(batch)*Q*D),k(size_t(batch)*K*D),v(k.size());
        std::vector<float> attended(q.size()),residual(x.size()),gate(size_t(batch)*F),up(gate.size());
        const Half* embeddings=tensor("model.embed_tokens.weight").half;
        for(int b=0;b<batch;++b)for(int j=0;j<H;++j)x[b*H+j]=embeddings[size_t(ids[b])*H+j];
        for(int layer=0;layer<L;++layer){std::string p="model.layers."+std::to_string(layer)+".";
            const float* nw=tensor(p+"input_layernorm.weight").norm;
            for(int b=0;b<batch;++b)norm(x.data()+b*H,nw,h.data()+b*H,H);
            project(tensor(p+"self_attn.q_proj.weight"),h.data(),q.data(),batch,pool);
            project(tensor(p+"self_attn.k_proj.weight"),h.data(),k.data(),batch,pool);
            project(tensor(p+"self_attn.v_proj.weight"),h.data(),v.data(),batch,pool);
            const float* qn=tensor(p+"self_attn.q_norm.weight").norm;const float* kn=tensor(p+"self_attn.k_norm.weight").norm;
            for(int b=0;b<batch;++b){const float* cs=cosines.data()+(pos+b)*64;const float* sn=sines.data()+(pos+b)*64;
                for(int head=0;head<Q;++head){float* a=q.data()+(b*Q+head)*D;norm(a,qn,a,D);rope(a,cs,sn);}
                for(int head=0;head<K;++head){float* a=k.data()+(b*K+head)*D;norm(a,kn,a,D);rope(a,cs,sn);}
                size_t dst=cache_offset(layer,pos+b);for(int j=0;j<K*D;++j){keys[dst+j]=Half(k[b*K*D+j]);values[dst+j]=Half(v[b*K*D+j]);}
            }
            pool.run(size_t(batch)*Q,[&](size_t begin,size_t end){
                std::vector<float> scores(pos+batch);
                for(size_t item=begin;item<end;++item){int b=item/Q,head=item%Q,kh=head/2,n=pos+b+1;
                    const float* query=q.data()+item*D;float maximum=-std::numeric_limits<float>::infinity();
                    for(int t=0;t<n;++t){float s=dot(query,keys.data()+cache_offset(layer,t,kh),D)*(1.0f/std::sqrt(float(D)));scores[t]=s;maximum=std::max(maximum,s);}
                    float sum=0;for(int t=0;t<n;++t){scores[t]=std::exp(scores[t]-maximum);sum+=scores[t];}
                    float* dest=attended.data()+item*D;std::fill(dest,dest+D,0.0f);
                    for(int t=0;t<n;++t){float prob=scores[t]/sum;const Half* val=values.data()+cache_offset(layer,t,kh);
                        for(int j=0;j<D;j+=8){float16x8_t vv=vld1q_f16(val+j);vst1q_f32(dest+j,vfmaq_n_f32(vld1q_f32(dest+j),vcvt_f32_f16(vget_low_f16(vv)),prob));vst1q_f32(dest+j+4,vfmaq_n_f32(vld1q_f32(dest+j+4),vcvt_f32_f16(vget_high_f16(vv)),prob));}
                    }
                }
            });
            project(tensor(p+"self_attn.o_proj.weight"),attended.data(),residual.data(),batch,pool);
            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];
            nw=tensor(p+"post_attention_layernorm.weight").norm;for(int b=0;b<batch;++b)norm(x.data()+b*H,nw,h.data()+b*H,H);
            project(tensor(p+"mlp.gate_proj.weight"),h.data(),gate.data(),batch,pool);project(tensor(p+"mlp.up_proj.weight"),h.data(),up.data(),batch,pool);
            for(size_t j=0;j<gate.size();++j)gate[j]=gate[j]*(0.5f+0.5f*std::tanh(gate[j]*0.5f))*up[j];
            project(tensor(p+"mlp.down_proj.weight"),gate.data(),residual.data(),batch,pool);
            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];
        }
        pos+=batch;
        // Keep only last hidden state; vocabulary is computed once per API call.
        output.resize(H);norm(x.data()+size_t(batch-1)*H,tensor("model.norm.weight").norm,output.data(),H);
    }
    void append(const std::vector<int>& ids,bool prefill) {
        require(!ids.empty(),"Empty token input");require(ids.size()<=size_t(options.context-pos),"Context capacity exceeded");
        for(int id:ids)require(id>=0&&id<vocabulary,"Token ID out of range");
        Pool& pool=prefill?*prefill_pool:*decode_pool;
        for(size_t begin=0;begin<ids.size();begin+=64){int batch=std::min(size_t(64),ids.size()-begin);chunk(ids.data()+begin,batch,pool);}
        auto hidden=std::move(output);output.resize(vocabulary);project(tensor("model.embed_tokens.weight"),hidden.data(),output.data(),1,pool);
    }
};
Model::Model(const std::string& path,Options options):impl_(std::make_unique<Impl>(path,options)){}
Model::~Model()=default;
void Model::reset(){impl_->pos=0;impl_->output.clear();}
void Model::prefill(const std::vector<int>& ids){impl_->append(ids,true);}
void Model::step(int token){impl_->append(std::vector<int>{token},false);}
const std::vector<float>& Model::logits()const{return impl_->output;}
int Model::position()const{return impl_->pos;}
}
