"""Transport policy tests: no neural arithmetic or timing."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
SOURCE=r'''
#include "phase_pool_policy.h"
#include <cassert>
#include <stdexcept>
using namespace phase_pool;
int main(){
 for(const char* selector:{"OFF","PREFILL_12"}){
  Policy p(selector);bool enabled=p.enabled;p.validate_config(8,8);
  for(int n=1;n<=8;n++){
   Policy::Scope scope(p,Purpose::prefill);
   assert(p.prefill_pool()==enabled);p.record();
  }
  assert(p.purpose==Purpose::scalar);
  for(int n=1;n<=4;n++)for(int logits=0;logits<2;logits++){
   Policy::Scope scope(p,Purpose::verification);
   assert(!p.prefill_pool());p.record();
  }
  for(int logits=0;logits<2;logits++){assert(!p.prefill_pool());p.record();}
  try{Policy::Scope scope(p,Purpose::prefill);throw std::runtime_error("tick failed");}catch(...){}
  assert(p.purpose==Purpose::scalar);
  assert(p.dispatches[0][2]==8 && p.dispatches[0][0]==2);
  assert(p.dispatches[enabled?1:0][1]==8);
  for(int n:{0,6,12,32}){bool failed=false;try{p.validate_config(n,8);}catch(...){failed=true;}assert(failed);}
  bool failed=false;try{p.validate_config(8,1);}catch(...){failed=true;}assert(failed);
  assert(p.enabled==enabled && p.purpose==Purpose::scalar);
 }
 for(const char* bad:{"12","ON","prefill_12","OFF "}){bool failed=false;try{Policy p(bad);}catch(...){failed=true;}assert(failed);}
 assert(!Policy(nullptr).enabled && !Policy("").enabled);
}
'''
class PhasePoolPolicyTests(unittest.TestCase):
 def test_policy(self):
  with tempfile.TemporaryDirectory()as directory:
   source=Path(directory)/'test.cpp';source.write_text(SOURCE);binary=Path(directory)/'test'
   subprocess.run(['c++','-std=c++17','-I'+str(ROOT/'sim'),str(source),'-o',str(binary)],check=True,capture_output=True)
   subprocess.run([str(binary)],check=True,capture_output=True)
if __name__=='__main__':unittest.main()
