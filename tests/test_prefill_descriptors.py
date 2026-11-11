"""Count-aware descriptor capacity remains bounded before scheduling."""
import subprocess
import tempfile
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class PrefillDescriptorTests(unittest.TestCase):
    def test_opt_in_capacity_and_span_validation(self):
        source=r'''
#include "projection_descriptors.h"
static int spans=0;
extern "C" void validate_weight_span(int64_t i,int64_t n) {
    ++spans;if(i<0 || n<1 || i+n>10000000)throw std::runtime_error("span");
}
int main() {
 std::array<ProjectionDescriptor,3> jobs{{{0,2048,1024,0},{0,1024,1024,2048},{0,1024,1024,3072}}};
 double x=0,y=0;
 for(int count=1;count<=9;++count) {
  bool accepted=true;try {validate_projection_group(jobs,3,count,&x,&y);}catch(...){accepted=false;}
#ifdef PREFILL_B8
  if(accepted!=(count<=8))return 1;
#else
  if(accepted!=(count<=4))return 2;
#endif
 }
 auto original=jobs;
 for(int bad=0;bad<4;++bad) {
  jobs=original;
  if(bad==0)jobs[1].output_offset=2049;
  if(bad==1)jobs[0].width=1025;
  if(bad==2)jobs[2].rows=4096;
  if(bad==3)jobs[0].base=10000000;
  try {validate_projection_group(jobs,3,4,&x,&y);return 3;}catch(...){}
 }
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'check.cpp').write_text(source)
            for defines in ([],['-DPREFILL_B8']):
                subprocess.run(['c++','-std=c++17',*defines,'-I'+str(ROOT/'sim'),str(path/'check.cpp'),'-o',str(path/'check')],check=True,capture_output=True)
                subprocess.run([str(path/'check')],check=True,capture_output=True)

if __name__=='__main__':unittest.main()
