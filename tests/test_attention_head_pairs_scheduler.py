"""Bounded head ownership, transport stride, eligibility and validation checks."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from tools.experiment import ROOT

@unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
class HeadPairScheduler(unittest.TestCase):
    def test_pair_partitions_and_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'verilated.h').write_text('#pragma once\nclass VerilatedContext{public:void threads(int){}};\n')
            (path/'svdpi.h').write_text('#pragma once\n')
            (path/'Vprojection_worker.h').write_text(r'''
#pragma once
#include "verilated.h"
#include <array>
#include <cstdint>
#include <stdexcept>
class Vprojection_worker {
public:
 int reduction_lanes=8,start=0,clk=0,width=0,first_row=0,end_row=0;
 int operation=0,attention_layer=0,attention_position=0;
 int64_t base=0;
 std::array<double,4096> activation{};
 std::array<double,151936> result{};
 explicit Vprojection_worker(VerilatedContext*){}
 void eval(){
  if(!start || !clk)return;
  if(base==999)throw std::runtime_error("injected partition failure");
  int stride=operation==6?256:128;
  for(int row=first_row;row<end_row;++row)
   for(int j=0;j<stride;++j)result[row*stride+j]=activation[row*stride+j]+attention_position;
 }
 void final(){}
};
''')
            (path/'test.cpp').write_text(r'''
#include "projection_pool.h"
#include <cassert>
extern "C" void validate_weight_span(int64_t,int64_t){throw std::runtime_error("attention must not read weights");}
int main(){
 for(int workers:{1,2,3,4,5,7,8,9,16,17,32}){
  ProjectionPool pool(workers,8);
  bool eligible=workers==1 || workers==2 || workers==4 || workers==8;
  assert(pool.head_pair_eligible()==eligible);
  double x[2048],y[2049];for(int i=0;i<2048;++i)x[i]=i*3;
  for(int position:{0,63,4095}){
   for(auto& value:y)value=-777;
   pool.run(0,eligible?8:16,2048,x,y,eligible?6:1,27,position);
   for(int i=0;i<2048;++i)assert(y[i]==x[i]+position);
   assert(y[2048]==-777);
  }
  if(!eligible){
   bool failed=false;try{pool.run(0,8,2048,x,y,6,0,0);}catch(const std::runtime_error&){failed=true;}assert(failed);
  }
  for(auto shape:{std::array<int,4>{-1,0,8,2048},{28,0,8,2048},{0,-1,8,2048},
                 {0,4096,8,2048},{0,0,16,2048},{0,0,8,1024}}){
   bool failed=false;
   try{pool.run(0,shape[2],shape[3],x,y,6,shape[0],shape[1]);}catch(const std::runtime_error&){failed=true;}
   assert(failed);
  }
  bool failed=false;
  try{pool.run(999,eligible?8:16,2048,x,y,eligible?6:1,0,0);}catch(const std::runtime_error&){failed=true;}
  assert(failed);pool.shutdown();pool.shutdown();
 }
}
''')
            for hybrid in (False,True):
                with self.subTest(hybrid=hybrid):
                    executable=path/'check'
                    subprocess.run(['c++','-std=c++17','-O2','-pthread','-DPARALLEL_ATTENTION',
                        '-DATTENTION_HEAD_PAIRS',*(['-DPROJECTION_HYBRID_WAIT']if hybrid else []),
                        '-I',str(path),'-I',str(ROOT/'sim'),str(path/'test.cpp'),'-o',str(executable)],
                        check=True,capture_output=True,timeout=30)
                    subprocess.run([str(executable)],check=True,capture_output=True,timeout=20)

if __name__=='__main__':unittest.main()
