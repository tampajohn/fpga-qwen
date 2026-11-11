"""Compile the actual transport definitions against exhaustive BF16 fixtures."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT/'sim/full_model.cpp').read_text()
transport = source[source.index('static const uint16_t* weights;'):source.index('extern "C" int64_t tensor_base')]
preamble = '''
#define WEIGHT_CACHE_F32
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <thread>
#include <atomic>
using svOpenArrayHandle = void*;
static void* svGetArrayPtr(void* p) { return p; }
'''
main = r'''
static uint64_t bits(double x) { uint64_t result; std::memcpy(&result,&x,8); return result; }
static double original(uint16_t x) { uint32_t b=uint32_t(x)<<16;float f;std::memcpy(&f,&b,4);return f; }
template<class F> static void must_throw(F fn) { bool threw=false;try{fn();}catch(const std::runtime_error&){threw=true;}if(!threw)throw std::runtime_error("Missing bounds rejection"); }
int main() {
 std::vector<uint16_t> input(65536);for(size_t i=0;i<input.size();++i)input[i]=i;
 weights=input.data();words=input.size();prepare_weight_cache();size_t finite=0,nans=0;
 for(size_t i=0;i<words;++i) {
  double a=original(input[i]),b=weight_value(i),c=weight_value_unchecked(i);
  if(std::isnan(a)){if(!std::isnan(b)||!std::isnan(c))throw std::runtime_error("NaN classification");++nans;}
  else {if(bits(a)!=bits(b)||bits(a)!=bits(c))throw std::runtime_error("Transport bits differ");if(std::isfinite(a))++finite;}
 }
 std::vector<double> row(3072);
 for(int count:{1,3072})for(size_t start:{size_t(0),words-size_t(count)}) {
  weight_row(start,count,row.data());for(int i=0;i<count;++i){double a=original(input[start+i]);if(!std::isnan(a)&&bits(a)!=bits(row[i]))throw std::runtime_error("Row differs");}
 }
 must_throw([]{weight_value(-1);});must_throw([]{weight_value(words);});
 must_throw([]{validate_weight_span(-1,1);});must_throw([]{validate_weight_span(0,-1);});
 must_throw([]{validate_weight_span(words,1);});must_throw([]{validate_weight_span(words-1,2);});
 must_throw([&]{weight_row(0,0,row.data());});must_throw([&]{weight_row(0,3073,row.data());});
 must_throw([&]{weight_row(words,1,row.data());});must_throw([]{weight_row(0,1,nullptr);});
 std::atomic<bool> bad{false};std::vector<std::thread> threads;
 for(size_t worker=0;worker<8;++worker)threads.emplace_back([&,worker]{for(size_t i=worker;i<words;i+=8){double a=original(input[i]);if(!std::isnan(a)&&bits(a)!=bits(weight_value_unchecked(i)))bad=true;}});
 for(auto& thread:threads)thread.join();if(bad)throw std::runtime_error("Concurrent transport differs");
 pid_t child=fork();if(child<0)throw std::runtime_error("fork failed");
 if(child==0){rlimit limit{0,0};setrlimit(RLIMIT_CORE,&limit);*const_cast<volatile float*>(expanded_weights)=1.0f;_exit(2);}
 int status=0;waitpid(child,&status,0);if(!WIFSIGNALED(status)||(WTERMSIG(status)!=SIGSEGV&&WTERMSIG(status)!=SIGBUS))throw std::runtime_error("Cache mapping writable");
 munmap(const_cast<float*>(expanded_weights),expanded_bytes);
 std::cout<<"PASS finite_patterns="<<finite<<" nan_patterns="<<nans<<" signed_zero/exact_rows/bounds/concurrent_reads/read_only\n";
}
'''
directory=ROOT/'build/weight-cache-transport'
directory.mkdir(parents=True,exist_ok=True)
cpp=directory/'probe.cpp';cpp.write_text(preamble+transport+main)
binary=directory/'probe'
subprocess.run(['c++','-std=c++17','-O3','-pthread',str(cpp),'-o',str(binary)],check=True)
subprocess.run([str(binary)],check=True)
