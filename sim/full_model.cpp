#ifdef PREFILL_B8
static constexpr int block_capacity=8;
#else
static constexpr int block_capacity=4;
#endif
// Transport-only harness: map checkpoint bytes, supply tensor addresses, clock
// Verilator, and exchange integer token IDs. No neural-network arithmetic here.
#if defined(FP32_FMA_CANONICAL_PREFIX) && !defined(FP32_PREFILL_FMA)
#error "Fused canonical prefix capability requires fused FP32 prefill"
#endif
#include "Vqwen_behavioral.h"
#include "Vqwen_behavioral___024root.h"
#include "Vqwen_behavioral__Dpi.h"
#include "verilated.h"
#ifdef FP32_KV
#if !defined(FP32_PROJECTION_B1) || !defined(FP32_VERIFICATION_B4) || !defined(PREFILL_B8) || !defined(PARALLEL_ATTENTION) || defined(FP32_PREFILL_B8) || defined(STATIC_KV)
#error "FP32 KV requires hybrid B1/B4/B8 parallel mode without competing KV/prefill layouts"
#endif
#include <type_traits>
#include "Vfp32_kv_store.h"
#include "Vfp32_kv_store___024root.h"
#endif
#include "projection_pool.h"
#ifdef PHASE_POOLS_EXPERIMENT
#include "phase_pool_policy.h"
#include <cstdlib>
#endif
#include <chrono>
#include <iomanip>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
static const uint16_t* weights;
static size_t words;
static std::vector<int64_t> offsets;
extern "C" double weight_value(int64_t index) {
    if (index<0 || size_t(index)>=words) throw std::runtime_error("Weight address out of bounds");
    uint32_t bits=uint32_t(weights[index])<<16;
    float value; std::memcpy(&value,&bits,4);
    return value;
}
// Projection jobs validate their complete span before any worker is dispatched.
// This accessor is used only by those validated SV worker loops.
extern "C" void validate_weight_span(int64_t index,int64_t count) {
    if(index<0 || count<0 || size_t(index)>words || size_t(count)>words-size_t(index))
        throw std::runtime_error("Projection weight span out of bounds");
}
extern "C" double weight_value_unchecked(int64_t index) {
    uint32_t bits=uint32_t(weights[index])<<16;
    float value;std::memcpy(&value,&bits,4);return value;
}
// Row transport variant retained for direct comparison with resident/scalar modes.
extern "C" void weight_row(int64_t index,int count,const svOpenArrayHandle output) {
    auto* destination=static_cast<double*>(svGetArrayPtr(output));
    if(!destination || index<0 || count<1 || count>3072 ||
       size_t(index)>words || size_t(count)>words-size_t(index))
        throw std::runtime_error("Weight row address out of bounds");
    for(int i=0;i<count;i++) {
        uint32_t bits=uint32_t(weights[index+i])<<16;
        float value;std::memcpy(&value,&bits,4);destination[i]=value;
    }
}
extern "C" int64_t tensor_base(int slot) { return offsets.at(slot); }
static std::unique_ptr<ProjectionPool> pool;
#ifdef PHASE_POOLS_EXPERIMENT
#if !defined(FP32_VERIFICATION_B4) || !defined(BATCH_ATTENTION) || !defined(PREFILL_B8)
#error "Phase pools require hybrid B4 verification and B8 batched attention"
#endif
static std::unique_ptr<ProjectionPool> prefill_pool;
static uint64_t phase_pool_constructions=0;
static std::unique_ptr<phase_pool::Policy> phase_policy;
#endif
static void shutdown_pools() {
#ifdef PHASE_POOLS_EXPERIMENT
    prefill_pool.reset();
#endif
    pool.reset();
}
struct PoolLifetime {~PoolLifetime(){shutdown_pools();}};
static bool verification_command=false;
static ProjectionPool* selected_pool() {
#ifdef PHASE_POOLS_EXPERIMENT
    if(verification_command!=(phase_policy->purpose==phase_pool::Purpose::verification))throw std::runtime_error("Inconsistent verification command purpose");
    return phase_policy->prefill_pool()?prefill_pool.get():pool.get();
#else
    return pool.get();
#endif
}
static void record_dispatch() {
#ifdef PHASE_POOLS_EXPERIMENT
    phase_policy->record();
#endif
}
// Only the coordinator changes command purpose; run() waits for all readers.
struct VerificationCommand {
    VerificationCommand() {verification_command=true;}
    ~VerificationCommand() {verification_command=false;}
};
#ifdef FP32_KV
static std::unique_ptr<VerilatedContext> kv_context;
static std::unique_ptr<Vfp32_kv_store> kv_store;
static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<Vfp32_kv_store___024root>().fp32_kv_store__DOT__keys[0])>,float>);
static_assert(std::is_same_v<std::remove_reference_t<decltype(std::declval<Vfp32_kv_store___024root>().fp32_kv_store__DOT__values[0])>,float>);
static_assert(sizeof(std::declval<Vfp32_kv_store___024root>().fp32_kv_store__DOT__keys)+sizeof(std::declval<Vfp32_kv_store___024root>().fp32_kv_store__DOT__values)==939524096);
extern "C" void write_fp32_kv(int layer,int position,const svOpenArrayHandle keys,const svOpenArrayHandle values) {
    const auto* k=static_cast<const double*>(svGetArrayPtr(keys));
    const auto* v=static_cast<const double*>(svGetArrayPtr(values));
    if(!kv_store || layer<0 || layer>=28 || position<0 || position>=4096 || !k || !v || svSize(keys,1)!=1024 || svSize(values,1)!=1024)
        throw std::runtime_error("Invalid FP32 KV write");
    // Coordinator is the sole writer; every pool dispatch is synchronous.
    kv_store->clk=0;kv_store->write_enable=1;kv_store->write_base=(layer*4096+position)*1024;
    std::memcpy(kv_store->write_keys.data(),k,1024*sizeof(double));
    std::memcpy(kv_store->write_values.data(),v,1024*sizeof(double));
    kv_store->eval();kv_store->clk=1;kv_store->eval();
}
#endif
#ifdef PARALLEL_ATTENTION
static const Vqwen_behavioral___024root* attention_root=nullptr;
extern "C" double attention_key_value(int index) {
    if(!attention_root || index<0 || index>=28*4096*1024)
        throw std::runtime_error("Invalid attention key address");
#ifdef FP32_KV
    return kv_store->rootp->fp32_kv_store__DOT__keys[index];
#elif defined(STATIC_KV)
    return attention_root->qwen_behavioral__DOT__keys[index];
#else
    return attention_root->__Vnontrivs.qwen_behavioral__DOT__keys.at(index);
#endif
}
extern "C" double attention_value_value(int index) {
    if(!attention_root || index<0 || index>=28*4096*1024)
        throw std::runtime_error("Invalid attention value address");
#ifdef FP32_KV
    return kv_store->rootp->fp32_kv_store__DOT__values[index];
#elif defined(STATIC_KV)
    return attention_root->qwen_behavioral__DOT__values[index];
#else
    return attention_root->__Vnontrivs.qwen_behavioral__DOT__values.at(index);
#endif
}
extern "C" void dispatch_attention(int layer,int position,const svOpenArrayHandle query,
                                    const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    const auto* q=static_cast<const double*>(svGetArrayPtr(query));
    auto* a=static_cast<double*>(svGetArrayPtr(output));
    if(!active_pool || !q || !a || !attention_root) throw std::runtime_error("Invalid attention dispatch");
    record_dispatch();
#ifdef ATTENTION_HEAD_PAIRS
    if(svSize(query,1)!=2048 || svSize(output,1)!=2048)
        throw std::runtime_error("Invalid paired-head attention storage");
    if(active_pool->head_pair_eligible())active_pool->run(0,8,2048,q,a,6,layer,position);
    else
#endif
    active_pool->run(0,16,2048,q,a,1,layer,position);
}
#ifdef BATCH_ATTENTION
extern "C" void dispatch_block_attention(int layer,int position,int count,
        const svOpenArrayHandle query,const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    if(svSize(query,1)!=2048*8 || svSize(output,1)!=2048*8)
        throw std::runtime_error("Invalid block attention shape");
    const auto* q=static_cast<const double*>(svGetArrayPtr(query));
    auto* a=static_cast<double*>(svGetArrayPtr(output));
    if(!active_pool || !q || !a || !attention_root)throw std::runtime_error("Invalid block attention dispatch");
    record_dispatch();
    active_pool->run(0,16,2048,q,a,5,layer,position,count);
}
#endif
// This guard is destroyed before the main model, including exception unwinding.
struct AttentionLifetime {
    ~AttentionLifetime() {shutdown_pools();attention_root=nullptr;}
};
#endif

extern "C" void dispatch_projection(int64_t base,int rows,int width,
                                    const svOpenArrayHandle input,const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    const auto* x=static_cast<const double*>(svGetArrayPtr(input));
    auto* y=static_cast<double*>(svGetArrayPtr(output));
    if(!active_pool || !x || !y) throw std::runtime_error("Invalid projection dispatch");
    record_dispatch();
    active_pool->run(base,rows,width,x,y);
}
#ifdef BATCH_PREFILL
extern "C" void dispatch_block_projection(int64_t base,int rows,int width,int count,
                                          const svOpenArrayHandle input,const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    const auto* x=static_cast<const double*>(svGetArrayPtr(input));
    auto* y=static_cast<double*>(svGetArrayPtr(output));
    if(!active_pool || !x || !y) throw std::runtime_error("Invalid block projection dispatch");
    record_dispatch();
    active_pool->run(base,rows,width,x,y,verification_command?4:(count>4?3:2),0,0,count);
}
#endif
#ifdef SCALAR_GROUP
extern "C" void dispatch_scalar_group(const svOpenArrayHandle bases,
        const svOpenArrayHandle rows,const svOpenArrayHandle output_offsets,int count,
        const svOpenArrayHandle input,const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    if(!active_pool || svSize(bases,1)!=3 || svSize(rows,1)!=3 || svSize(output_offsets,1)!=3 ||
       svSize(input,1)!=1024 || svSize(output,1)!=6144)
        throw std::runtime_error("Invalid scalar group array shape");
    const auto* b=static_cast<const int64_t*>(svGetArrayPtr(bases));
    const auto* r=static_cast<const int*>(svGetArrayPtr(rows));
    const auto* o=static_cast<const int*>(svGetArrayPtr(output_offsets));
    const auto* x=static_cast<const double*>(svGetArrayPtr(input));
    auto* y=static_cast<double*>(svGetArrayPtr(output));
    if(!b || !r || !o || !x || !y || count<2 || count>3)
        throw std::runtime_error("Invalid scalar group storage/count");
    std::array<ProjectionDescriptor,3> jobs{};
    for(int i=0;i<count;++i)jobs[i]={b[i],r[i],1024,o[i]};
    record_dispatch();
    active_pool->run_group(jobs,count,1,x,y,true);
}
#endif

#ifdef GROUP_PROJECTIONS
extern "C" void dispatch_projection_group(const svOpenArrayHandle bases,
        const svOpenArrayHandle rows,const svOpenArrayHandle output_offsets,int count,int tokens,
        const svOpenArrayHandle input,const svOpenArrayHandle output) {
    auto* active_pool=selected_pool();
    if(!active_pool || svSize(bases,1)!=3 || svSize(rows,1)!=3 || svSize(output_offsets,1)!=3 ||
       svSize(input,1)<1024*block_capacity || svSize(output,1)!=6144*block_capacity)
        throw std::runtime_error("Invalid group array shape");
    const auto* b=static_cast<const int64_t*>(svGetArrayPtr(bases));
    const auto* r=static_cast<const int*>(svGetArrayPtr(rows));
    const auto* o=static_cast<const int*>(svGetArrayPtr(output_offsets));
    const auto* x=static_cast<const double*>(svGetArrayPtr(input));
    auto* y=static_cast<double*>(svGetArrayPtr(output));
    if(!b || !r || !o || !x || !y) throw std::runtime_error("Invalid group array storage");
    if(count<2 || count>3) throw std::runtime_error("Invalid projection group count");
    std::array<ProjectionDescriptor,3> jobs{};
    for(int i=0;i<count;++i) jobs[i]={b[i],r[i],1024,o[i]};
    record_dispatch();
    active_pool->run_group(jobs,count,tokens,x,y,false,verification_command);
}
#endif

int main(int argc,char** argv) {
    try {
        PoolLifetime early_pool_lifetime;
        if(argc!=3 && argc!=4) throw std::runtime_error("usage: Vqwen_behavioral checkpoint.safetensors offsets.txt [workers: 0=serial]");
        int fd=open(argv[1],O_RDONLY);
        struct stat info{};
        if(fd<0 || fstat(fd,&info)) throw std::runtime_error("Cannot open checkpoint");
        void* mapped=mmap(nullptr,info.st_size,PROT_READ,MAP_PRIVATE,fd,0);
        if(mapped==MAP_FAILED) throw std::runtime_error("Cannot map checkpoint");
        close(fd);
        weights=static_cast<const uint16_t*>(mapped); words=info.st_size/2;
        std::ifstream layout(argv[2]); int64_t offset;
        while(layout>>offset) offsets.push_back(offset);
        if(offsets.size()!=311) throw std::runtime_error("Invalid tensor layout");
        int count=argc==4?std::stoi(argv[3]):0;
        if(count<0 || count>32) throw std::runtime_error("Invalid worker count");
#ifdef FP32_PROJECTION_B1
        if(count==0) throw std::runtime_error("FP32 scalar projections require workers");
        int reduction_lanes=8;
#else
        int reduction_lanes=1;
#endif
#ifdef PHASE_POOLS_EXPERIMENT
        phase_policy=std::make_unique<phase_pool::Policy>(std::getenv("QWEN_EXPERIMENTAL_PHASE_POOLS"));
        phase_policy->validate_config(count,reduction_lanes);
#endif
        if(count) pool=std::make_unique<ProjectionPool>(count,reduction_lanes);
#ifdef PHASE_POOLS_EXPERIMENT
        ++phase_pool_constructions;
        prefill_pool=std::make_unique<ProjectionPool>(12,reduction_lanes);
        ++phase_pool_constructions;
#endif
        auto context=std::make_unique<VerilatedContext>();
#ifdef FP32_KV
        kv_context=std::make_unique<VerilatedContext>();kv_context->threads(1);
        kv_store=std::make_unique<Vfp32_kv_store>(kv_context.get());
        kv_store->clk=0;kv_store->write_enable=0;kv_store->read_index=0;kv_store->eval();
#endif
        auto d=std::make_unique<Vqwen_behavioral>(context.get());
#ifdef PARALLEL_ATTENTION
        attention_root=d->rootp;
        AttentionLifetime attention_lifetime;
#endif
        auto tick=[&](){d->clk=0;d->eval();d->clk=1;d->eval();};
#ifdef BATCH_PREFILL
        d->block_start=0;
#endif
        d->sample_enabled=0;d->sample_uniform=0.0;d->presence_penalty=0.0;
        d->parallel_workers=count; d->rst_n=0; d->start=0; d->truncate_cache=0; tick(); d->rst_n=1;
        std::cout<<"ready\n"<<std::flush;
        std::string command;
        while(std::cin>>command) {
            if(command=="quit") break;
            if(command=="reset") {
                d->start=0;d->rst_n=0;tick();d->rst_n=1;
                std::cout<<"reset\n"<<std::flush;
            } else if(command=="truncate") {
                int length;
                if(!(std::cin>>length) || length<0 || length>d->position)
                    throw std::runtime_error("Invalid cache length");
                d->start=0;d->truncate_cache=1;d->truncate_to=length;tick();d->truncate_cache=0;
                std::cout<<"truncated\n"<<std::flush;
            } else if(command=="lanes") {
                int requested;
                if(!(std::cin>>requested) || (requested!=1 && requested!=8))
                    throw std::runtime_error("Reduction lanes must be 1 or 8");
#ifdef FP32_PROJECTION_B1
                if(requested!=8) throw std::runtime_error("FP32 scalar projections require eight reduction lanes");
#endif
#ifdef PHASE_POOLS_EXPERIMENT
                phase_policy->validate_config(count,requested);
#else
                pool.reset();reduction_lanes=requested;
                if(count) pool=std::make_unique<ProjectionPool>(count,reduction_lanes);
#endif
                std::cout<<"configured\n"<<std::flush;
            } else if(command=="workers") {
                int requested;
                if(!(std::cin>>requested) || requested<0 || requested>32)
                    throw std::runtime_error("Invalid worker count");
#ifdef FP32_PROJECTION_B1
                if(requested==0) throw std::runtime_error("FP32 scalar projections require workers");
#endif
#ifdef PHASE_POOLS_EXPERIMENT
                phase_policy->validate_config(requested,reduction_lanes);
#else
                pool.reset();
                if(requested) pool=std::make_unique<ProjectionPool>(requested,reduction_lanes);
                count=requested;d->parallel_workers=count;
#endif
                std::cout<<"configured\n"<<std::flush;
#ifdef BATCH_PREFILL
#ifdef PREFILL_B8
            } else if(command=="prefill") {
                int count;
                if(!(std::cin>>count) || count<1 || count>8 || d->position+count>4096)
                    throw std::runtime_error("Invalid prefill request");
                std::array<int,8> tokens{};
                for(int t=0;t<count;++t)
                    if(!(std::cin>>tokens[t]) || tokens[t]<0 || tokens[t]>=151936)
                        throw std::runtime_error("Invalid prefill token");
#ifdef PHASE_POOLS_EXPERIMENT
                phase_pool::Policy::Scope prefill_scope(*phase_policy,phase_pool::Purpose::prefill);
#endif
                for(int t=0;t<8;++t) {d->block_tokens[t]=tokens[t];d->block_uniforms[t]=0.0;}
                d->block_count=count;d->do_logits=0;d->sample_enabled=0;
                d->start=0;d->block_start=1;tick();d->block_start=0;
                if(!d->done) throw std::runtime_error("Prefill simulation did not finish");
                std::cout<<"prefill "<<count<<' '<<d->position<<'\n'<<std::flush;
#endif
            } else if(command=="block") {
#if defined(FP32_PROJECTION_B1) && !defined(FP32_VERIFICATION_B4)
                throw std::runtime_error("FP32 B1 hybrid does not support block verification; use FP64 prefill");
#else
                int count,logits,sampled;
                double penalty;
                if(!(std::cin>>count>>logits>>sampled>>penalty) || count<1 || count>4 ||
                   (logits!=0 && logits!=1) || (sampled!=0 && sampled!=1) ||
                   !(penalty>=0 && penalty<=2) || d->position+count>4096)
                    throw std::runtime_error("Invalid block request");
                for(int t=0;t<count;++t) {
                    int token;
                    if(!(std::cin>>token) || token<0 || token>=151936)
                        throw std::runtime_error("Invalid block token");
                    d->block_tokens[t]=token;
                }
                for(int t=0;t<count;++t) {
                    double uniform;
                    if(!(std::cin>>uniform) || !(uniform>=0 && uniform<1))
                        throw std::runtime_error("Invalid block uniform");
                    d->block_uniforms[t]=uniform;
                }
                d->block_count=count;d->do_logits=logits;d->sample_enabled=sampled;
#ifdef FP32_VERIFICATION_B4
                VerificationCommand verification_scope;
#ifdef PHASE_POOLS_EXPERIMENT
                phase_pool::Policy::Scope phase_verification_scope(*phase_policy,phase_pool::Purpose::verification);
#endif
#endif
                d->presence_penalty=penalty;d->start=0;d->block_start=1;tick();d->block_start=0;
                if(!d->done) throw std::runtime_error("Block simulation did not finish");
                std::cout<<"block "<<count<<std::setprecision(17);
                for(int t=0;t<count;++t)
                    std::cout<<' '<<d->block_ids[t]<<' '<<d->block_positions[t]<<' '<<d->block_logits[t];
                std::cout<<std::setprecision(6)<<'\n'<<std::flush;
#endif // FP32_PROJECTION_B1
#endif
            } else if(command=="token" || command=="sample") {
                int token, logits;
                if(!(std::cin>>token>>logits)) throw std::runtime_error("Malformed token command");
                d->sample_enabled=command=="sample";
                if(d->sample_enabled) {
                    double uniform,penalty;
                    if(!(std::cin>>uniform>>penalty) || !(uniform>=0 && uniform<1) || !(penalty>=0 && penalty<=2))
                        throw std::runtime_error("Invalid sampling parameters");
                    d->sample_uniform=uniform;d->presence_penalty=penalty;
                }
                d->token_id=token;d->do_logits=logits;d->start=1;tick();d->start=0;
                if(!d->done) throw std::runtime_error("Simulation did not finish token");
                std::cout<<d->best_id<<' '<<d->position<<' '<<d->best_logit<<'\n'<<std::flush;
#ifdef GROUP_PROJECTIONS
            } else if(command=="groupstats") {
                if(!pool) throw std::runtime_error("Group statistics require workers");
                auto values=pool->group_stats();
#ifdef PHASE_POOLS_EXPERIMENT
                const auto extra=prefill_pool->group_stats();
                for(int i=0;i<4;i++)values[i]+=extra[i];
#endif
                std::cout<<values[0]<<' '<<values[1]<<' '<<values[2]<<' '<<values[3]<<'\n'<<std::flush;
#endif
#ifdef ATTENTION_QUERY_VALUE_DISPATCH
            } else if(command=="query_value_policy") {
                const auto& policy=query_value::configured_policy();
                std::array<uint64_t,2> counts{};
                if(pool) counts=pool->query_value_stats();
#ifdef PHASE_POOLS_EXPERIMENT
                const auto extra=prefill_pool->query_value_stats();
                for(int i=0;i<2;i++)counts[i]+=extra[i];
#endif
                std::cout<<"{\"selector\":\""<<policy.selector<<"\",\"min_context\":"<<policy.min_context
                    <<",\"context_metric\":\"paired_query_common_kv_tokens\",\"requires_second_query\":true,\"paired_head_pairs\":"<<counts[0]
                    <<",\"fallback_head_pairs\":"<<counts[1]<<"}\n"<<std::flush;
#endif
#ifdef FP32_PROJECTION_B1
#ifdef PHASE_POOLS_EXPERIMENT
            } else if(command=="phase_pools") {
                std::cout<<"{\"selector\":\""<<(phase_policy->enabled?"PREFILL_12":"OFF")<<"\",\"resident_decode_workers\":8,\"resident_prefill_workers\":12,\"prefill_workers\":"<<(phase_policy->enabled?12:8)<<",\"decode_workers\":8,\"pool_constructions\":"<<phase_pool_constructions<<",\"dispatch_counts\":[";
                for(int p=0;p<2;p++){if(p)std::cout<<',';std::cout<<'[';for(int k=0;k<3;k++){if(k)std::cout<<',';std::cout<<phase_policy->dispatches[p][k];}std::cout<<']';}
                std::cout<<"],\"group_counters\":[";
                const std::array<std::array<uint64_t,4>,2> groups{pool->group_stats(),prefill_pool->group_stats()};
                for(int p=0;p<2;p++){if(p)std::cout<<',';std::cout<<'[';for(int k=0;k<4;k++){if(k)std::cout<<',';std::cout<<groups[p][k];}std::cout<<']';}
                std::cout<<"]}"<<'\n'<<std::flush;
#endif
            } else if(command=="precision") {
#ifdef FP32_KV
                std::cout<<R"({"precision_profile":"fp32_b1_fp64_prefill_fp32_kv","scalar_projection":"fp32","prefill_projection":"fp64","verification_projection":"fp32","attention_arithmetic":"fp64","kv_storage":"fp32","max_verification_block":4,"prefix_reuse_supported":false,"canonical_prefix_policy":"none"})"<<'\n'<<std::flush;
#elif defined(FP32_PREFILL_FMA)
#ifdef FP32_FMA_CANONICAL_PREFIX
                std::cout<<R"({"precision_profile":"fp32_b1_fp32_fma_prefill","scalar_projection":"fp32","prefill_projection":"fp32","verification_projection":"fp32","vocabulary_projection":"fp32","scalar_accumulation":"separate","prefill_accumulation":"fused","verification_accumulation":"separate","vocabulary_accumulation":"separate","attention_kv":"fp64","max_verification_block":4,"prefix_reuse_supported":false,"canonical_prefix_policy":"fp32_fma_b8_aligned_v1"})"<<'\n'<<std::flush;
#else
                std::cout<<R"({"precision_profile":"fp32_b1_fp32_fma_prefill","scalar_projection":"fp32","prefill_projection":"fp32","verification_projection":"fp32","vocabulary_projection":"fp32","scalar_accumulation":"separate","prefill_accumulation":"fused","verification_accumulation":"separate","vocabulary_accumulation":"separate","attention_kv":"fp64","max_verification_block":4,"prefix_reuse_supported":false,"canonical_prefix_policy":"none"})"<<'\n'<<std::flush;
#endif
#elif defined(FP32_PREFILL_B8)
                std::cout<<R"({"precision_profile":"fp32_b1_fp32_prefill","scalar_projection":"fp32","prefill_projection":"fp32","verification_projection":"fp32","attention_kv":"fp64","max_verification_block":4,"prefix_reuse_supported":false,"canonical_prefix_policy":"none"})"<<'\n'<<std::flush;
#elif defined(FP32_VERIFICATION_B4)
                std::cout<<R"({"precision_profile":"fp32_b1_fp64_prefill","scalar_projection":"fp32","prefill_projection":"fp64","verification_projection":"fp32","attention_kv":"fp64","max_verification_block":4,"prefix_reuse_supported":false,"canonical_prefix_policy":"fp64_b8_aligned_v1"})"<<'\n'<<std::flush;
#else
                std::cout<<R"({"precision_profile":"fp32_b1_fp64_prefill","scalar_projection":"fp32","prefill_projection":"fp64","attention_kv":"fp64","max_verification_block":1,"prefix_reuse_supported":false,"canonical_prefix_policy":"fp64_b8_aligned_v1"})"<<'\n'<<std::flush;
#endif
            } else if(command=="fp32probe") {
                int slot,rows,width;
                std::string path;
                if(!(std::cin>>slot>>rows>>width>>path) || !pool || slot<0 ||
                   size_t(slot)>=offsets.size() || rows<1 || rows>151936 ||
                   width<8 || width>3072 || width%8)
                    throw std::runtime_error("Invalid FP32 projection probe");
                const int64_t base=tensor_base(slot);
                validate_weight_span(base,int64_t(rows)*width);
                // Synthetic fixture preparation only; all projection math is SV.
                std::vector<double> input(width),output(rows);
                for(int j=0;j<width;++j) input[j]=double((j*7)%61-30)/17.0;
                pool->run(base,rows,width,input.data(),output.data());
                std::ofstream out(path,std::ios::binary);
                out.write(reinterpret_cast<const char*>(output.data()),rows*sizeof(double));
                out.close();
                if(!out) throw std::runtime_error("Cannot write FP32 probe outputs");
                std::cout<<"probed\n"<<std::flush;
#endif
            } else if(command=="bench") {
                int slot,rows,width,repetitions;
                if(!(std::cin>>slot>>rows>>width>>repetitions) || !pool || repetitions<1 || repetitions>1000)
                    throw std::runtime_error("Invalid benchmark request");
                std::vector<double> input(width),output(rows);
                for(int i=0;i<width;i++) input[i]=double(i%31-15)/16;
                pool->run(tensor_base(slot),rows,width,input.data(),output.data());
                auto begin=std::chrono::steady_clock::now();
                for(int i=0;i<repetitions;i++) pool->run(tensor_base(slot),rows,width,input.data(),output.data());
                double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
                std::cout<<std::setprecision(17)<<seconds<<' '<<output.front()<<' '<<output.back()<<'\n'<<std::flush;
            } else if(command=="dump") {
                std::string path;std::cin>>path;
                std::ofstream out(path,std::ios::binary);
                auto& logits=d->rootp->qwen_behavioral__DOT__logits;
                out.write(reinterpret_cast<const char*>(logits.data()),151936*sizeof(double));
                // The acknowledgement is the reader's synchronization boundary.
                // Close (including buffered tail bytes) before publishing it.
                out.close();
                if(!out) throw std::runtime_error("Cannot write logits");
                std::cout<<"dumped\n"<<std::flush;
            } else throw std::runtime_error("Unknown command");
        }
        d->final();shutdown_pools();munmap(mapped,info.st_size);
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
