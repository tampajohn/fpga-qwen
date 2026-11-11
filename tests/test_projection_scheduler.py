"""Exercise pool publication, wakeups and errors using a small transport mock.

The mock is not model inference and makes no numerical-quality assertion.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

from tools.experiment import ROOT


@unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
class ProjectionSchedulerTests(unittest.TestCase):
    def test_cv_and_hybrid_publication_and_failure_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'verilated.h').write_text(
                '#pragma once\nclass VerilatedContext {public: void threads(int) {}};\n')
            (path / 'svdpi.h').write_text('#pragma once\n')
            (path / 'Vprojection_worker.h').write_text(r'''
#pragma once
#include "verilated.h"
#include <array>
#include <cstdint>
#include <atomic>
extern std::atomic<int> worker_evaluations;
#include <stdexcept>
class Vprojection_worker {
public:
    int reduction_lanes=1,start=0,clk=0,width=0,first_row=0,end_row=0;
    int operation=0,attention_layer=0,attention_position=0,batch_count=1;
    int64_t base=0;
    std::array<double,3072> activation{};
    std::array<double,151936> result{};
    std::array<double,12288> batch_activation{},batch_result{};
    explicit Vprojection_worker(VerilatedContext*) {}
    void eval() {
        if(!start || !clk) return;
        ++worker_evaluations;
        if(operation==2) {
            for(int row=first_row;row<end_row;++row)
                for(int t=0;t<4;++t) batch_result[row*4+t]=batch_activation[(row%width)*4+t]+base;
            return;
        }
        if(operation==1) {
            for(int head=first_row;head<end_row;++head)
                for(int j=0;j<128;++j)
                    result[head*128+j]=activation[head*128+j]+attention_layer+attention_position;
            return;
        }
        if(base==999) throw std::runtime_error("injected worker failure");
        for(int row=first_row;row<end_row;++row)
            result[row]=activation[row%width]+base;
    }
    void final() {}
};
''')
            (path / 'test.cpp').write_text(r'''
#include "projection_pool.h"
#include <cassert>
#include <chrono>
std::atomic<int> worker_evaluations{0};
extern "C" void validate_weight_span(int64_t index,int64_t count) {
    if(index<0 || count<0 || index>8000000 || count>8000000-index) throw std::runtime_error("invalid span");
}
int main() {
    for(int count : {1,3,8}) {
        ProjectionPool pool(count,8);
        double x[17],y[101];
        for(int run=0;run<300;++run) {
            for(int i=0;i<17;++i) x[i]=run*100+i;
            if(run%50==0) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            pool.run(run,101,17,x,y);
            for(int row=0;row<101;++row) assert(y[row]==x[row%17]+run);
        }
        bool caught=false;
        try {pool.run(-1,101,17,x,y);} catch(const std::runtime_error&) {caught=true;}
        assert(caught);
        for(auto shape : {std::pair<int,int>{0,17}, {151937,17}, {101,0}, {101,3073}}) {
            caught=false;
            try {pool.run(1,shape.first,shape.second,x,y);} catch(const std::runtime_error&) {caught=true;}
            assert(caught);
        }
        pool.run(1,101,17,x,y);
        caught=false;
        try {pool.run(999,101,17,x,y);} catch(const std::runtime_error&) {caught=true;}
        assert(caught);
        pool.shutdown();pool.shutdown();
    }
#ifdef PARALLEL_ATTENTION
    for(int count : {1,3,8,17}) {
        ProjectionPool pool(count);
        double q[2048], a[2048];
        for(int i=0;i<2048;++i) q[i]=i;
        for(int cycle=0;cycle<4;++cycle) {
            pool.run(0,16,2048,q,a,1,2,cycle);
            for(int i=0;i<2048;++i) assert(a[i]==i+2+cycle);
            pool.run(3,101,17,q,a);
            for(int i=0;i<101;++i) assert(a[i]==q[i%17]+3);
        }
    }
#endif

#ifdef GROUP_PROJECTIONS
    for(int threads : {1,3,17}) {
        ProjectionPool pool(threads,8);
        double input[4096],output[24576];
        for(int i=0;i<4096;++i) input[i]=i;
        std::array<ProjectionDescriptor,3> valid{{{100,7,1024,0},{200,5,1024,7},{300,3,1024,12}}};
        for(int tokens : {1,4}) {
            pool.run_group(valid,3,tokens,input,output);
            for(const auto& job:valid) for(int row=0;row<job.rows;++row) for(int t=0;t<tokens;++t)
                assert(output[(job.output_offset+row)*4+t]==input[row*4+t]+job.base);
        }
        pool.run_group(valid,3,1,input,output,true);
        for(const auto& job:valid) for(int row=0;row<job.rows;++row)
            assert(output[job.output_offset+row]==input[row]+job.base);
        {
            int before=worker_evaluations.load();auto counters=pool.group_stats();bool caught=false;
            try {pool.run_group(valid,3,4,input,output,true);}catch(const std::runtime_error&){caught=true;}
            assert(caught);assert(worker_evaluations.load()==before);assert(counters==pool.group_stats());
        }
        auto reject=[&](std::array<ProjectionDescriptor,3> jobs,int count=3,int tokens=4,
                        const double* x=nullptr,double* y=nullptr) {
            int before=worker_evaluations.load();auto counters=pool.group_stats();bool caught=false;
            try {pool.run_group(jobs,count,tokens,x?x:input,y?y:output);} catch(const std::runtime_error&) {caught=true;}
            assert(caught);assert(worker_evaluations.load()==before);assert(counters==pool.group_stats());
        };
        reject(valid,1);reject(valid,4);reject(valid,3,0);reject(valid,3,5);
        for(bool null_input : {false,true}) {
            int before=worker_evaluations.load();bool caught=false;
            try {pool.run_group(valid,3,4,null_input?nullptr:input,null_input?output:nullptr);}
            catch(const std::runtime_error&) {caught=true;}
            assert(caught);assert(worker_evaluations.load()==before);
        }
        auto bad=valid;bad[1].output_offset=8;reject(bad);
        bad=valid;bad[1].output_offset=6;reject(bad);
        bad=valid;bad[1].width=2048;reject(bad);
        bad=valid;bad[0].rows=0;reject(bad);
        bad=valid;bad[0].rows=3073;reject(bad);
        bad={{{100,3072,1024,0},{200,3072,1024,3072},{300,1,1024,6144}}};reject(bad);
        bad=valid;bad[2].base=8000000;reject(bad);
        bad=valid;bad[0].base=-1;reject(bad);
        // A normal job after a group must not replay stale descriptors.
        pool.run(5,8,1024,input,output,2,0,0,4);
        for(int row=0;row<8;++row) for(int t=0;t<4;++t) assert(output[row*4+t]==input[row*4+t]+5);
    }
#endif
    for(int i=0;i<10;++i) {ProjectionPool idle(3);}
}
''')
            for hybrid, attention, grouped in ((False,False,False),(True,False,False),(False,True,False),(True,True,False),(False,True,True),(True,True,True)):
                with self.subTest(hybrid=hybrid, attention=attention, grouped=grouped):
                    executable = path / ('hybrid' if hybrid else 'cv')
                    flags = (['-DGROUP_PROJECTIONS','-DBATCH_PREFILL'] if grouped else []) + (['-DPARALLEL_ATTENTION'] if attention else []) + (['-DPROJECTION_HYBRID_WAIT'] if hybrid else [])
                    subprocess.run(['c++', '-std=c++17', '-O2', '-pthread', *flags,
                                    '-I', str(path), '-I', str(ROOT / 'sim'),
                                    str(path / 'test.cpp'), '-o', str(executable)],
                                   check=True, capture_output=True, timeout=30)
                    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
