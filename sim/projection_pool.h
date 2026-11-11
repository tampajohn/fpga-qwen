// Persistent CPU workers, each running an independent Verilated SV partition.
// Host code only schedules partitions and copies activations/results.
#pragma once
#include "Vprojection_worker.h"
#include "verilated.h"
#include "svdpi.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <exception>
#include <cstring>
#include <stdexcept>

class ProjectionPool {
    std::mutex mutex;
    std::condition_variable work, complete;
    std::vector<std::thread> threads;
    bool stopping=false;
    uint64_t generation=0;
    int remaining=0, ready=0;
    std::exception_ptr error;
    int64_t base=0;
    int rows=0, width=0;
    const double* input=nullptr;
    double* output=nullptr;
    void worker(int id, int count) {
        try {
            auto context=std::make_unique<VerilatedContext>();
            context->threads(1);
            auto model=std::make_unique<Vprojection_worker>(context.get());
            model->start=0;model->clk=0;model->eval();
            std::unique_lock<std::mutex> lock(mutex);
            ++ready;complete.notify_one();
            uint64_t seen=0;
            while(true) {
                work.wait(lock,[&]{return stopping || generation!=seen;});
                if(stopping) break;
                seen=generation;
                int begin=rows*id/count, end=rows*(id+1)/count;
                model->base=base;model->width=width;model->first_row=begin;model->end_row=end;
                std::memcpy(model->activation.data(),input,width*sizeof(double));
                double* destination=output;
                lock.unlock();
                model->start=1;model->clk=0;model->eval();model->clk=1;model->eval();
                std::memcpy(destination+begin,model->result.data()+begin,(end-begin)*sizeof(double));
                lock.lock();
                if(--remaining==0) complete.notify_one();
            }
            lock.unlock();model->final();
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            error=std::current_exception();stopping=true;
            complete.notify_one();work.notify_all();
        }
    }
public:
    explicit ProjectionPool(int count) {
        if(count<1 || count>32) throw std::runtime_error("Workers must be 1..32");
        for(int i=0;i<count;i++) threads.emplace_back([this,i,count]{worker(i,count);});
        std::unique_lock<std::mutex> lock(mutex);
        complete.wait(lock,[&]{return ready==count || error;});
        if(error) {
            lock.unlock();shutdown();std::rethrow_exception(error);
        }
    }
    void shutdown() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;work.notify_all();}
        for(auto& thread:threads) if(thread.joinable()) thread.join();
    }
    ~ProjectionPool() {shutdown();}
    void run(int64_t matrix_base,int matrix_rows,int matrix_width,const double* x,double* y) {
        if(matrix_rows<1 || matrix_rows>151936 || matrix_width<1 || matrix_width>3072)
            throw std::runtime_error("Invalid projection shape");
        std::unique_lock<std::mutex> lock(mutex);
        if(error) std::rethrow_exception(error);
        base=matrix_base;rows=matrix_rows;width=matrix_width;input=x;output=y;
        remaining=threads.size();++generation;work.notify_all();
        complete.wait(lock,[&]{return remaining==0 || error;});
        if(error) std::rethrow_exception(error);
    }
};
