"""Transport-only hybrid routing, descriptor partition and failure lifecycle checks."""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from tools.experiment import ROOT


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
class FP32SchedulerTests(unittest.TestCase):
    def test_partition_routing_groups_and_failure_join(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'verilated.h').write_text('#pragma once\nclass VerilatedContext {public: void threads(int) {}};\n')
            (path/'svdpi.h').write_text('#pragma once\n')
            template=r'''
#pragma once
#include "verilated.h"
#include <array>
#include <cstdint>
#include <stdexcept>
class MODEL {
public:
 int reduction_lanes=8,start=0,clk=0,width=0,first_row=0,end_row=0;
 int operation=0,attention_layer=0,attention_position=0,batch_count=1;
 int64_t base=0;
 std::array<double,4096> activation{};
 std::array<double,151936> result{};
 std::array<double,12288> batch_activation{},batch_result{};
 std::array<double,24576> wide_activation{},wide_result{};
 explicit MODEL(VerilatedContext*) {}
 void eval() {
  if(!start || !clk)return;
  if(base==999)throw std::runtime_error("injected worker error");
  for(int row=first_row;row<end_row;++row) {
   if(operation==1) {
    for(int j=0;j<128;++j)result[row*128+j]=activation[row*128+j]+MARKER;
   } else if(operation==2) {
    for(int t=0;t<4;++t)batch_result[row*4+t]=batch_activation[(row%width)*4+t]+base+MARKER;
   } else if(operation==3) {
    for(int t=0;t<8;++t)wide_result[row*8+t]=wide_activation[(row%width)*8+t]+base+MARKER;
   } else result[row]=activation[row%width]+base+MARKER;
  }
 }
 void final() {}
};
'''
            for model,marker in [('Vprojection_worker',1000),('Vfp32_projection_kernel',2000)]:
                (path/(model+'.h')).write_text(template.replace('MODEL',model).replace('MARKER',str(marker)))
            (path/'test.cpp').write_text(r'''
#include "projection_pool.h"
#include <cassert>
extern "C" void validate_weight_span(int64_t base,int64_t count) {
 if(base<0 || count<0 || base+count>8000000)throw std::runtime_error("invalid span");
}
int main() {
 for(int threads:{1,3,17}) {
  ProjectionPool pool(threads,8);
  double x[8192],y[49152];
  for(int i=0;i<8192;++i)x[i]=i;
  for(int cycle=0;cycle<3;++cycle) {
   pool.run(5,23,16,x,y);
   for(int r=0;r<23;++r)assert(y[r]==x[r%16]+2005);
   pool.run(0,16,2048,x,y,1,0,0);
   for(int r=0;r<2048;++r)assert(y[r]==x[r]+1000);
   for(int tokens:{1,4,5,8}) {
    pool.run(7,23,16,x,y,tokens>4?3:2,0,0,tokens);
    for(int r=0;r<23;++r)for(int t=0;t<tokens;++t)assert(y[r*8+t]==x[(r%16)*8+t]+1007);
   }
  }
  std::vector<double> vocabulary(151937,-777.0);
  pool.run(11,151936,8,x,vocabulary.data());
  for(int r=0;r<151936;++r)assert(vocabulary[r]==x[r%8]+2011);
  assert(vocabulary[151936]==-777.0);
  std::array<ProjectionDescriptor,3> jobs{{{10,7,1024,0},{20,5,1024,7},{30,3,1024,12}}};
  pool.run_group(jobs,3,1,x,y,true);
  for(auto job:jobs)for(int r=0;r<job.rows;++r)assert(y[job.output_offset+r]==x[r]+job.base+2000);
  auto stats=pool.group_stats();assert(stats[0]==1 && stats[1]==3 && stats[2]==2);
  assert(stats[3]==uint64_t(2)*threads*1024*sizeof(double));
  for(int tokens:{1,4,8}) {
   pool.run_group(jobs,3,tokens,x,y);
   for(auto job:jobs)for(int r=0;r<job.rows;++r)for(int t=0;t<tokens;++t)
    assert(y[(job.output_offset+r)*8+t]==x[r*8+t]+job.base+1000);
  }
  pool.run(8,23,16,x,y);
  for(int r=0;r<23;++r)assert(y[r]==x[r%16]+2008);
  bool caught=false;
  try{pool.run(8,23,9,x,y);}catch(const std::runtime_error&){caught=true;}assert(caught);
  caught=false;
  try{pool.run(999,23,16,x,y);}catch(const std::runtime_error&){caught=true;}assert(caught);
  pool.shutdown();pool.shutdown();
 }
 for(auto shape:{std::pair<int,int>{0,8},{1,1}}) {
  bool caught=false;try{ProjectionPool invalid(shape.first,shape.second);}catch(const std::runtime_error&){caught=true;}assert(caught);
 }
}
''')
            flags=['-DFP32_PROJECTION_B1','-DPARALLEL_ATTENTION','-DBATCH_PREFILL','-DPREFILL_B8','-DGROUP_PROJECTIONS']
            for hybrid in (False,True):
                with self.subTest(hybrid_wait=hybrid):
                    binary=path/'check'
                    subprocess.run(['c++','-std=c++17','-O2','-pthread',*flags,
                        *(['-DPROJECTION_HYBRID_WAIT'] if hybrid else []),'-I',str(path),'-I',str(ROOT/'sim'),
                        str(path/'test.cpp'),'-o',str(binary)],check=True,capture_output=True,timeout=30)
                    subprocess.run([str(binary)],check=True,capture_output=True,timeout=20)


if __name__=='__main__':unittest.main()
