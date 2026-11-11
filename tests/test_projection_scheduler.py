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
#include <stdexcept>
class Vprojection_worker {
public:
    int reduction_lanes=1,start=0,clk=0,width=0,first_row=0,end_row=0;
    int64_t base=0;
    std::array<double,3072> activation{};
    std::array<double,151936> result{};
    explicit Vprojection_worker(VerilatedContext*) {}
    void eval() {
        if(!start || !clk) return;
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
extern "C" void validate_weight_span(int64_t index,int64_t count) {
    if(index<0 || count<0) throw std::runtime_error("invalid span");
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
    for(int i=0;i<10;++i) {ProjectionPool idle(3);}
}
''')
            for hybrid in (False, True):
                with self.subTest(hybrid=hybrid):
                    executable = path / ('hybrid' if hybrid else 'cv')
                    flags = ['-DPROJECTION_HYBRID_WAIT'] if hybrid else []
                    subprocess.run(['c++', '-std=c++17', '-O2', '-pthread', *flags,
                                    '-I', str(path), '-I', str(ROOT / 'sim'),
                                    str(path / 'test.cpp'), '-o', str(executable)],
                                   check=True, capture_output=True, timeout=30)
                    subprocess.run([str(executable)], check=True, capture_output=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
