// Transport-only harness: map checkpoint bytes, supply tensor addresses, clock
// Verilator, and exchange integer token IDs. No neural-network arithmetic here.
#include "Vqwen_behavioral.h"
#include "Vqwen_behavioral___024root.h"
#include "verilated.h"
#include "projection_pool.h"
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
extern "C" void dispatch_projection(int64_t base,int rows,int width,
                                    const svOpenArrayHandle input,const svOpenArrayHandle output) {
    const auto* x=static_cast<const double*>(svGetArrayPtr(input));
    auto* y=static_cast<double*>(svGetArrayPtr(output));
    if(!pool || !x || !y) throw std::runtime_error("Invalid projection dispatch");
    pool->run(base,rows,width,x,y);
}
int main(int argc,char** argv) {
    try {
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
        int reduction_lanes=1;
        if(count) pool=std::make_unique<ProjectionPool>(count,reduction_lanes);
        auto context=std::make_unique<VerilatedContext>();
        auto d=std::make_unique<Vqwen_behavioral>(context.get());
        auto tick=[&](){d->clk=0;d->eval();d->clk=1;d->eval();};
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
                pool.reset();reduction_lanes=requested;
                if(count) pool=std::make_unique<ProjectionPool>(count,reduction_lanes);
                std::cout<<"configured\n"<<std::flush;
            } else if(command=="workers") {
                int requested;
                if(!(std::cin>>requested) || requested<0 || requested>32)
                    throw std::runtime_error("Invalid worker count");
                pool.reset();
                if(requested) pool=std::make_unique<ProjectionPool>(requested,reduction_lanes);
                count=requested;d->parallel_workers=count;
                std::cout<<"configured\n"<<std::flush;
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
                if(!out) throw std::runtime_error("Cannot write logits");
                std::cout<<"dumped\n"<<std::flush;
            } else throw std::runtime_error("Unknown command");
        }
        d->final();pool.reset();munmap(mapped,info.st_size);
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
