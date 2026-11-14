#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <iostream>
class Pool {
    struct alignas(128) Completion { std::atomic<uint64_t> generation{0}; };
    int n_; std::vector<std::thread> threads_; std::mutex mutex_;
    std::condition_variable start_; std::atomic<bool> stop_{false};
    alignas(128) std::atomic<uint64_t> generation_{0};
    std::array<Completion,64> completed_{};
    int parked_=0; size_t count_=0; int active_=0;
    void* job_context_=nullptr;
    void (*job_)(void*,size_t,size_t,int)=nullptr; std::exception_ptr error_;
    void work(int i) {
        uint64_t seen=0;
        for(;;) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::microseconds(50);
            unsigned spins=0;
            while(generation_.load(std::memory_order_acquire)==seen && !stop_.load(std::memory_order_relaxed)) {
                // ARM's scheduling hint does not enter the kernel.
                __asm__ volatile("yield" ::: "memory");
                if((++spins&63)==0 && std::chrono::steady_clock::now()>=deadline)break;
            }
            if(stop_.load(std::memory_order_acquire))return;
            if(generation_.load(std::memory_order_acquire)==seen) {
                std::unique_lock lock(mutex_);++parked_;
                start_.wait(lock,[&]{return stop_.load(std::memory_order_relaxed) || generation_.load(std::memory_order_acquire)!=seen;});
                --parked_;
                if(stop_.load(std::memory_order_relaxed))return;
            }
            const uint64_t next=generation_.load(std::memory_order_acquire);
            const size_t count=count_;
            try {if(i<active_)job_(job_context_,count*i/active_,count*(i+1)/active_,i);}
            catch(...) {std::lock_guard lock(mutex_);if(!error_)error_=std::current_exception();}
            seen=next;completed_[i].generation.store(seen,std::memory_order_release);
        }
    }
public:
    explicit Pool(int n):n_(n) {
        try {for(int i=1;i<n_;++i)threads_.emplace_back([this,i]{work(i);});}
        catch(...) {shutdown();throw;}
    }
    ~Pool(){shutdown();}
    int worker_count()const{return n_;}
    void shutdown() {
        {std::lock_guard lock(mutex_);stop_.store(true,std::memory_order_release);}
        start_.notify_all();for(auto& t:threads_)if(t.joinable())t.join();
    }
    template<class Fn> void run(size_t count,Fn fn) { run_active(count,n_,fn); }
    template<class Fn> void run_active(size_t count,int active,Fn fn) {
        if(active<1 || active>n_)throw std::invalid_argument("Invalid active worker count");
        uint64_t next;bool wake;
        {
            std::lock_guard lock(mutex_);
            // fn stays on this caller's stack until every completion is acquired.
            // A non-owning thunk avoids per-kernel std::function allocations.
            job_context_=&fn;job_=[](void* context,size_t begin,size_t end,int worker){(*static_cast<Fn*>(context))(begin,end,worker);};
            count_=count;active_=active;error_=nullptr;
            next=generation_.load(std::memory_order_relaxed)+1;
            generation_.store(next,std::memory_order_release);wake=parked_!=0;
        }
        if(wake)start_.notify_all();
        try {job_(job_context_,0,count/active_,0);}
        catch(...) {std::lock_guard lock(mutex_);if(!error_)error_=std::current_exception();}
        for(int i=1;i<n_;++i) {
            unsigned spins=0;
            while(completed_[i].generation.load(std::memory_order_acquire)!=next) {
                __asm__ volatile("yield" ::: "memory");
                if((++spins&1023)==0)std::this_thread::yield();
            }
        }
        job_=nullptr;job_context_=nullptr;if(error_)std::rethrow_exception(error_);
    }
};

int main(){
 try{
  size_t jobs=0;
  for(int workers:{1,6,8,12,64}){
   Pool pool(workers);
   for(int round=0;round<100;++round){
    if(round%10==0)std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for(size_t count:{size_t(0),size_t(1),size_t(7),size_t(16),size_t(17),size_t(151936)}){
     int active=(round%2)?std::min(8,workers):workers;
     std::array<size_t,64> begins{},ends{};std::array<int,64> calls{};
     pool.run_active(count,active,[&](size_t begin,size_t end,int worker){begins[worker]=begin;ends[worker]=end;++calls[worker];});
     for(int worker=0;worker<workers;++worker){
      if(worker<active){if(calls[worker]!=1||begins[worker]!=count*worker/active||ends[worker]!=count*(worker+1)/active)throw std::runtime_error("partition");}
      else if(calls[worker])throw std::runtime_error("inactive callback");
     }
     ++jobs;
    }
    bool caught=false;
    try{pool.run_active(16,std::min(workers,8),[&](size_t,size_t,int worker){if(worker==(round%2?0:std::min(workers,8)-1))throw std::runtime_error("expected");});}
    catch(const std::runtime_error&){caught=true;}
    if(!caught)throw std::runtime_error("exception lost");
    pool.run(1,[](size_t,size_t,int){}); // reuse after callback exception
    for(int invalid:{0,workers+1}){caught=false;try{pool.run_active(1,invalid,[](size_t,size_t,int){});}catch(const std::invalid_argument&){caught=true;}if(!caught)throw std::runtime_error("invalid count");}
   }
  }
  std::cout<<"{\"completed\":true,\"passed\":true,\"partition_jobs\":"<<jobs<<",\"workers\":[1,6,8,12,64]}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
