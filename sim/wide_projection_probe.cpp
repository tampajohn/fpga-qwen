// Isolated projection microbenchmark; arithmetic stays in SystemVerilog.
#include "Vprojection_worker.h"
#include "verilated.h"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static const uint16_t* weights;
static size_t words;
extern "C" double weight_value_unchecked(int64_t index) {
    uint32_t bits=uint32_t(weights[index])<<16;float value;
    std::memcpy(&value,&bits,4);return value;
}
extern "C" double weight_value(int64_t index) {
    if(index<0 || size_t(index)>=words) throw std::runtime_error("Invalid weight address");
    return weight_value_unchecked(index);
}
extern "C" double attention_key_value(int) {throw std::runtime_error("Unexpected attention");}
extern "C" double attention_value_value(int) {throw std::runtime_error("Unexpected attention");}
struct Result {double seconds;std::vector<double> values;};
static Result run(int64_t base,int rows,int width,int workers,int repetitions,int operation,int token_offset) {
    int tokens=operation==3?WIDE_BATCH:4;
    Result result{0,std::vector<double>(rows*tokens)};
    std::mutex mutex;std::condition_variable condition;int ready=0,done=0;bool go=false;
    std::vector<std::thread> threads;
    for(int worker=0;worker<workers;++worker) threads.emplace_back([&,worker] {
        auto context=std::make_unique<VerilatedContext>();context->threads(1);
        auto model=std::make_unique<Vprojection_worker>(context.get());
        model->start=0;model->clk=0;model->eval();
        model->operation=operation;model->reduction_lanes=8;model->batch_count=4;
        model->base=base;model->width=width;
        model->first_row=rows*worker/workers;model->end_row=rows*(worker+1)/workers;
        // Deterministic input fixture, not host model arithmetic.
        for(int column=0;column<width;++column) for(int t=0;t<tokens;++t) {
            #ifdef WIDE_FOUR_LANES
            double value=double((column*17+(token_offset+t)*13)%31-15)/17;
#else
            double value=double((column*17+(token_offset+t)*13)%31-15)/16;
#endif
            if(operation==3) model->wide_activation[column*WIDE_BATCH+t]=value;
            else model->batch_activation[column*4+t]=value;
        }
        auto tick=[&] {model->start=1;model->clk=0;model->eval();model->clk=1;model->eval();};
        tick(); // per-thread initialization and warmup excluded from timed region
        {std::unique_lock<std::mutex> lock(mutex);++ready;condition.notify_all();condition.wait(lock,[&]{return go;});}
        for(int i=0;i<repetitions;++i) tick();
        {std::lock_guard<std::mutex> lock(mutex);++done;condition.notify_all();}
        const double* source=operation==3?model->wide_result.data():model->batch_result.data();
        std::memcpy(result.values.data()+model->first_row*tokens,source+model->first_row*tokens,
                    (model->end_row-model->first_row)*tokens*sizeof(double));
        model->final();
    });
    std::unique_lock<std::mutex> lock(mutex);condition.wait(lock,[&]{return ready==workers;});
    auto start=std::chrono::steady_clock::now();go=true;condition.notify_all();
    condition.wait(lock,[&]{return done==workers;});
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    lock.unlock();for(auto& thread:threads) thread.join();return result;
}
int main(int argc,char** argv) {
    try {
        if(argc!=8) throw std::runtime_error("checkpoint offsets slot rows width workers repetitions");
        int fd=open(argv[1],O_RDONLY);struct stat info{};
        if(fd<0 || fstat(fd,&info)) throw std::runtime_error("Checkpoint open failed");
        void* mapping=mmap(nullptr,info.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
        if(mapping==MAP_FAILED) throw std::runtime_error("Checkpoint mmap failed");
        weights=static_cast<const uint16_t*>(mapping);words=info.st_size/2;
        std::ifstream file(argv[2]);std::vector<int64_t> offsets;int64_t offset;
        while(file>>offset) offsets.push_back(offset);
        int slot=std::stoi(argv[3]),rows=std::stoi(argv[4]),width=std::stoi(argv[5]);
        int workers=std::stoi(argv[6]),repetitions=std::stoi(argv[7]);
        if(rows<1 || rows>3072 || width<1 || width>3072 || workers<1 || workers>32 || repetitions<1 || repetitions>10000)
            throw std::runtime_error("Invalid shape/worker/repetition count");
        int64_t base=offsets.at(slot),elements=int64_t(rows)*width;
        if(base<0 || size_t(base)>words || size_t(elements)>words-size_t(base)) throw std::runtime_error("Invalid full span");
        auto wide=run(base,rows,width,workers,1,3,0);
        double maximum_error=0;bool bit_identical=true;
        for(int chunk=0;chunk<WIDE_BATCH/4;++chunk) {
            auto reference=run(base,rows,width,workers,1,2,chunk*4);
            for(int row=0;row<rows;++row) for(int t=0;t<4;++t) {
                double a=reference.values[row*4+t],b=wide.values[row*WIDE_BATCH+chunk*4+t];
                if(!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("Nonfinite output");
                maximum_error=std::max(maximum_error,std::abs(a-b));
                bit_identical=bit_identical && std::memcmp(&a,&b,sizeof(double))==0;
            }
        }
#ifdef WIDE_FOUR_LANES
        if(maximum_error>1e-10) throw std::runtime_error("Projection tolerance exceeded");
#else
        if(!bit_identical) throw std::runtime_error("Full projection output bits differ");
#endif
        std::cout<<std::setprecision(17)<<"{\"bit_identical\":"<<(bit_identical?"true":"false")
                 <<",\"max_absolute_error\":"<<maximum_error<<",\"batch\":"<<WIDE_BATCH<<",\"pairs\":[";
        for(int pair=0;pair<3;++pair) {
            double reference_time=0,wide_time=0;
            auto reference=[&] {reference_time=run(base,rows,width,workers,repetitions,2,0).seconds;};
            auto candidate=[&] {wide_time=run(base,rows,width,workers,repetitions,3,0).seconds;};
            if(pair%2) {candidate();reference();} else {reference();candidate();}
            if(pair) std::cout<<',';
            std::cout<<"{\"baseline_b4_seconds\":"<<reference_time<<",\"wide_seconds\":"<<wide_time
                     <<",\"throughput_ratio\":"<<(reference_time*(WIDE_BATCH/4))/wide_time<<'}';
        }
        std::cout<<"]}\n";munmap(mapping,info.st_size);
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
