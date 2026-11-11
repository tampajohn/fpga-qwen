// Persistent CPU workers, each running an independent Verilated SV partition.
// Host code only schedules partitions and copies activations/results.
// One coordinator owns run()/shutdown(); concurrent calls are not supported.
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
#include <atomic>
#include <array>

#ifdef GROUP_PROJECTIONS
#include "projection_descriptors.h"
#endif

extern "C" void validate_weight_span(int64_t index,int64_t count);

class ProjectionPool {
    std::mutex mutex;
    std::condition_variable work, complete;
    std::vector<std::thread> threads;
    std::atomic<bool> stopping{false};
#ifdef PROJECTION_HYBRID_WAIT
    std::atomic<uint64_t> published{0};
    std::atomic<int> pending{0};
    // Briefly remain runnable between adjacent layers, then sleep when idle.
    static void spin_hint() {
#if defined(__aarch64__)
        __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
        __asm__ __volatile__("pause");
#else
        std::this_thread::yield();
#endif
    }
    static constexpr int spin_limit=10000;
#endif
    uint64_t generation=0;
    int remaining=0, ready=0;
    std::exception_ptr error;
    int64_t base=0;
    int rows=0, width=0;
    int reduction_lanes=1;
#ifdef PARALLEL_ATTENTION
    int operation=0, attention_layer=0, attention_position=0;
#ifdef BATCH_PREFILL
    int batch_count=1;
#endif
#endif
#ifdef GROUP_PROJECTIONS
    std::array<ProjectionDescriptor,3> group_jobs{};
    int group_count=0;
    std::array<uint64_t,4> group_counters{};
#endif
    const double* input=nullptr;
    double* output=nullptr;
    void worker(int id, int count) {
        try {
            auto context=std::make_unique<VerilatedContext>();
            context->threads(1);
            auto model=std::make_unique<Vprojection_worker>(context.get());
            model->reduction_lanes=reduction_lanes;
            model->start=0;model->clk=0;model->eval();
            std::unique_lock<std::mutex> lock(mutex);
            ++ready;complete.notify_one();
            uint64_t seen=0;
#ifdef PROJECTION_HYBRID_WAIT
            lock.unlock();
#endif
            while(true) {
#ifdef PROJECTION_HYBRID_WAIT
                for(int spin=0;spin<spin_limit && !stopping.load(std::memory_order_acquire)
                    && published.load(std::memory_order_acquire)==seen;++spin) spin_hint();
                if(published.load(std::memory_order_acquire)==seen && !stopping.load()) {
                    lock.lock();
                    work.wait(lock,[&]{return stopping.load() || published.load(std::memory_order_acquire)!=seen;});
                    lock.unlock();
                }
                if(stopping.load(std::memory_order_acquire)) break;
                seen=published.load(std::memory_order_acquire);
#else
                work.wait(lock,[&]{return stopping || generation!=seen;});
                if(stopping) break;
                seen=generation;
#endif
#ifdef PARALLEL_ATTENTION
                model->operation=operation;
                model->attention_layer=attention_layer;model->attention_position=attention_position;
                int copy_scale=operation==1?128:1;
#ifdef BATCH_PREFILL
                model->batch_count=batch_count;
                if(operation==2) copy_scale=4;
#ifdef PREFILL_B8
                if(operation==3) copy_scale=8;
#endif
#endif
#else
                int copy_scale=1;
#endif
#ifdef BATCH_PREFILL
                #ifdef PREFILL_B8
                if(operation==3) std::memcpy(model->wide_activation.data(),input,width*8*sizeof(double));
                else if(operation==2) {
                    for(int j=0;j<width;++j) for(int t=0;t<4;++t)
                        model->batch_activation[j*4+t]=t<batch_count?input[j*8+t]:0.0;
                }
#else
                if(operation==2) std::memcpy(model->batch_activation.data(),input,width*4*sizeof(double));
#endif
                else
#endif
                std::memcpy(model->activation.data(),input,width*sizeof(double));
                double* destination=output;
#ifndef PROJECTION_HYBRID_WAIT
                lock.unlock();
#endif
#ifdef GROUP_PROJECTIONS
                for(int job=0;job<(group_count?group_count:1);++job) {
                    int job_rows=group_count?group_jobs[job].rows:rows;
                    int64_t job_base=group_count?group_jobs[job].base:base;
                    int output_offset=group_count?group_jobs[job].output_offset:0;
#else
                    int job_rows=rows;
                    int64_t job_base=base;
                    int output_offset=0;
#endif
                int begin=job_rows*id/count, end=job_rows*(id+1)/count;
                model->base=job_base;model->width=width;model->first_row=begin;model->end_row=end;
                model->start=1;model->clk=0;model->eval();model->clk=1;model->eval();
                const double* source=model->result.data();
#ifdef BATCH_PREFILL
                if(operation==2) source=model->batch_result.data();
#ifdef PREFILL_B8
                if(operation==3) source=model->wide_result.data();
#endif
#endif
                #ifdef PREFILL_B8
                if(operation==2) {
                    for(int r=begin;r<end;++r) for(int t=0;t<batch_count;++t)
                        destination[(output_offset+r)*8+t]=source[r*4+t];
                } else
#endif
                std::memcpy(destination+(output_offset+begin)*copy_scale,source+begin*copy_scale,
                            (end-begin)*copy_scale*sizeof(double));
#ifdef GROUP_PROJECTIONS
                }
#endif
#ifdef PROJECTION_HYBRID_WAIT
                // The release sequence carries every partition's writes to run().
                bool last=pending.fetch_sub(1,std::memory_order_acq_rel)==1;
                if(last) {
                    lock.lock();complete.notify_one();lock.unlock();
                }
#else
                lock.lock();
                if(--remaining==0) complete.notify_one();
#endif
            }
#ifndef PROJECTION_HYBRID_WAIT
            lock.unlock();
#endif
            model->final();
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);
            if(!error) error=std::current_exception();
            stopping=true;
            complete.notify_one();work.notify_all();
        }
    }
public:
    explicit ProjectionPool(int count,int lanes=1) : reduction_lanes(lanes) {
        if(lanes!=1 && lanes!=8) throw std::runtime_error("Reduction lanes must be 1 or 8");
        if(count<1 || count>32) throw std::runtime_error("Workers must be 1..32");
        try {
            for(int i=0;i<count;i++) threads.emplace_back([this,i,count]{worker(i,count);});
        } catch(...) {
            shutdown();throw;
        }
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
    void run(int64_t matrix_base,int matrix_rows,int matrix_width,const double* x,double* y
#ifdef PARALLEL_ATTENTION
             ,int requested_operation=0,int layer=0,int position=0
#ifdef BATCH_PREFILL
             ,int requested_batch_count=1
#endif
#endif
             ) {
#ifdef PARALLEL_ATTENTION
        if(requested_operation==1) {
            if(layer<0 || layer>=28 || position<0 || position>=4096 || matrix_rows!=16 || matrix_width!=2048)
                throw std::runtime_error("Invalid attention job");
        }
#ifdef BATCH_PREFILL
        else if(requested_operation==2) {
            if(requested_batch_count<1 || requested_batch_count>4 || matrix_rows>3072)
                throw std::runtime_error("Invalid block projection job");
        }
#endif
        #ifdef PREFILL_B8
        else if(requested_operation==3) {
            if(requested_batch_count<5 || requested_batch_count>8 || matrix_rows>3072)
                throw std::runtime_error("Invalid wide prefill projection job");
        }
#endif
        else if(requested_operation!=0) throw std::runtime_error("Invalid worker operation");
#endif
        if(matrix_rows<1 || matrix_rows>151936 || matrix_width<1 || matrix_width>3072)
            throw std::runtime_error("Invalid projection shape");
#ifdef PARALLEL_ATTENTION
        if(requested_operation!=1)
#endif
        validate_weight_span(matrix_base,int64_t(matrix_rows)*matrix_width);
        std::unique_lock<std::mutex> lock(mutex);
        if(error) std::rethrow_exception(error);
#ifdef PARALLEL_ATTENTION
        operation=requested_operation;attention_layer=layer;attention_position=position;
#ifdef BATCH_PREFILL
        batch_count=requested_batch_count;
#endif
#endif
#ifdef GROUP_PROJECTIONS
        group_count=0;
#endif
        base=matrix_base;rows=matrix_rows;width=matrix_width;input=x;output=y;
#ifdef PROJECTION_HYBRID_WAIT
        pending.store(threads.size(),std::memory_order_relaxed);
        published.fetch_add(1,std::memory_order_release);
        work.notify_all();
        lock.unlock();
        for(int spin=0;spin<spin_limit && pending.load(std::memory_order_acquire)!=0
            && !stopping.load(std::memory_order_acquire);++spin) spin_hint();
        lock.lock();
        complete.wait(lock,[&]{return pending.load(std::memory_order_acquire)==0 || error;});
#else
        remaining=threads.size();++generation;work.notify_all();
        complete.wait(lock,[&]{return remaining==0 || error;});
#endif
        if(error) {
            // A failing partition must not leave others writing caller-owned y
            // after this call returns by exception.
            auto failure=error;
            lock.unlock();shutdown();std::rethrow_exception(failure);
        }
    }
#ifdef GROUP_PROJECTIONS
    std::array<uint64_t,4> group_stats() const { return group_counters; }
    // Bounded immutable descriptors; every job reads the same normalized input.
    // Matrix reductions still execute in the existing SystemVerilog worker.
    void run_group(const std::array<ProjectionDescriptor,3>& jobs,int count,int tokens,
                   const double* x,double* y,bool scalar=false) {
        if(scalar && tokens!=1)throw std::runtime_error("Scalar group requires one token");
        validate_projection_group(jobs,count,tokens,x,y);
        std::unique_lock<std::mutex> lock(mutex);
        if(error) std::rethrow_exception(error);
        ++group_counters[0];group_counters[1]+=count;group_counters[2]+=count-1;
        group_counters[3]+=uint64_t(count-1)*threads.size()*1024*(scalar?1:(tokens>4?8:4))*sizeof(double);
        group_jobs=jobs;group_count=count;operation=scalar?0:(tokens>4?3:2);batch_count=tokens;
        base=jobs[0].base;rows=jobs[0].rows;width=1024;input=x;output=y;
#ifdef PROJECTION_HYBRID_WAIT
        pending.store(threads.size(),std::memory_order_relaxed);
        published.fetch_add(1,std::memory_order_release);
        work.notify_all();lock.unlock();
        for(int spin=0;spin<spin_limit && pending.load(std::memory_order_acquire)!=0
            && !stopping.load(std::memory_order_acquire);++spin) spin_hint();
        lock.lock();
        complete.wait(lock,[&]{return pending.load(std::memory_order_acquire)==0 || error;});
#else
        remaining=threads.size();++generation;work.notify_all();
        complete.wait(lock,[&]{return remaining==0 || error;});
#endif
        if(error) {auto failure=error;lock.unlock();shutdown();std::rethrow_exception(failure);}
    }
#endif

};
