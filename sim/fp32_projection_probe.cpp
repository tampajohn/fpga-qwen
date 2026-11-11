// Transport/timing only. Neural arithmetic and FP32 conversion execute in SV.
#include "verilated.h"
#ifdef BASELINE_FP64
#include "Vprojection_worker.h"
using Model=Vprojection_worker;
#else
#include "Vfp32_projection_kernel.h"
using Model=Vfp32_projection_kernel;
#endif
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <stdexcept>
static const uint16_t* weights;
static size_t words;
extern "C" double weight_value_unchecked(int64_t index){
    uint32_t bits=uint32_t(weights[index])<<16;float value;std::memcpy(&value,&bits,4);return value;
}
extern "C" double attention_key_value(int){throw std::runtime_error("Attention unsupported in projection probe");}
extern "C" double attention_value_value(int){throw std::runtime_error("Attention unsupported in projection probe");}
int main(int argc,char** argv){
 try{
    if(argc!=9)throw std::runtime_error("checkpoint offsets slot rows width workers reps output required");
    int fd=open(argv[1],O_RDONLY);struct stat info{};
    if(fd<0 || fstat(fd,&info))throw std::runtime_error("Cannot open checkpoint");
    void* mapping=mmap(nullptr,info.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(mapping==MAP_FAILED)throw std::runtime_error("Cannot map checkpoint");
    weights=static_cast<uint16_t*>(mapping);words=info.st_size/2;
    std::ifstream offsets_file(argv[2]);std::vector<int64_t> offsets;int64_t offset;
    while(offsets_file>>offset)offsets.push_back(offset);
    int slot=std::stoi(argv[3]),rows=std::stoi(argv[4]),width=std::stoi(argv[5]);
    int workers=std::stoi(argv[6]),reps=std::stoi(argv[7]);
    if(slot<0 || slot>=int(offsets.size()) || rows<1 || rows>KERNEL_ROWS || width<8 || width>3072 || width%8 || workers<1 || workers>32 || reps<1 || reps>10000)
        throw std::runtime_error("Invalid probe shape");
    const int64_t base=offsets[slot],span=int64_t(rows)*width;
    if(base<0 || size_t(base)>words || size_t(span)>words-size_t(base))throw std::runtime_error("Weight span invalid");
    constexpr int B=KERNEL_BATCH;
    std::vector<double> output(rows*B);
    std::mutex mutex;std::condition_variable cv;int ready=0,done=0;bool go=false;std::exception_ptr error;
    std::vector<std::thread> threads;
    try {for(int id=0;id<workers;id++)threads.emplace_back([&,id]{
      try{
        auto context=std::make_unique<VerilatedContext>();
        auto owner=std::make_unique<Model>(context.get());auto& model=*owner;
        model.base=base;model.width=width;model.first_row=rows*id/workers;model.end_row=rows*(id+1)/workers;
#ifdef BASELINE_FP64
        model.reduction_lanes=8;model.batch_count=B;model.operation=B==1?0:(B==4?2:3);
#endif
        for(int j=0;j<width;j++)for(int t=0;t<B;t++){
            const double input=double((j*7+t*11)%61-30)/17.0;
#ifdef BASELINE_FP64
            if(B==1)model.activation[j]=input;
            else if(B==4)model.batch_activation[j*4+t]=input;
            else model.wide_activation[j*8+t]=input;
#else
            model.activation[j*B+t]=input;
#endif
        }
        auto tick=[&]{model.start=1;model.clk=0;model.eval();model.clk=1;model.eval();};
        tick(); // Untimed warmup and model/input initialization.
        {std::unique_lock<std::mutex> lock(mutex);++ready;cv.notify_all();cv.wait(lock,[&]{return go;});}
        for(int r=0;r<reps;r++)tick();
        {std::lock_guard<std::mutex> lock(mutex);++done;cv.notify_all();}
        for(int row=model.first_row;row<model.end_row;row++)for(int t=0;t<B;t++){
#ifdef BASELINE_FP64
            output[row*B+t]=B==1?model.result[row]:(B==4?model.batch_result[row*4+t]:model.wide_result[row*8+t]);
#else
            output[row*B+t]=model.result[row*B+t];
#endif
        }
        model.final();
      }catch(...){std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();cv.notify_all();}
    });} catch(...) {
        {std::lock_guard<std::mutex> lock(mutex);go=true;cv.notify_all();}
        for(auto& thread:threads)if(thread.joinable())thread.join();throw;
    }
    std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return ready==workers || error;});
    auto start=std::chrono::steady_clock::now();go=true;cv.notify_all();
    cv.wait(lock,[&]{return done==workers || error;});auto end=std::chrono::steady_clock::now();lock.unlock();
    for(auto& thread:threads)thread.join();if(error)std::rethrow_exception(error);
    std::ofstream dump(argv[8],std::ios::binary);dump.write(reinterpret_cast<char*>(output.data()),output.size()*8);dump.close();
    if(!dump)throw std::runtime_error("Output write failed");
    std::cout<<std::setprecision(17)<<std::chrono::duration<double>(end-start).count()<<'\n';munmap(mapping,info.st_size);
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
