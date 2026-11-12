#include "Vfp32_kv_fixture.h"
#include "Vfp32_kv_fixture___024root.h"
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
// Transport exact binary64 words; no float conversion or neural arithmetic here.
static double decode(uint64_t bits) { double result; std::memcpy(&result,&bits,8); return result; }
static uint64_t encode(double value) { uint64_t result; std::memcpy(&result,&value,8); return result; }
int main() {
    Vfp32_kv_fixture model;
    model.clk=0; model.write_enable=0; model.read_index=0; model.eval();
    int enabled,index,read_index; uint64_t key,value;
    while(std::cin >> enabled >> index >> read_index >> std::hex >> key >> value >> std::dec) {
        model.clk=0; model.write_enable=enabled; model.write_index=index;
        model.write_key=decode(key); model.write_value=decode(value); model.read_index=read_index; model.eval();
        model.clk=1; model.eval();
        // A future read-only callback may widen resident float storage exactly.
        // It never calls a worker model or narrows/writes on the host.
        static_assert(std::is_same_v<std::remove_reference_t<decltype(model.rootp->fp32_kv_fixture__DOT__keys[0])>, float>);
        if(model.read_valid) {
            const double key_read=model.rootp->fp32_kv_fixture__DOT__keys[read_index];
            const double value_read=model.rootp->fp32_kv_fixture__DOT__values[read_index];
            if(encode(key_read)!=encode(model.read_key) || encode(value_read)!=encode(model.read_value))
                throw std::runtime_error("rootp widening disagrees with SV real read ports");
        }
        std::cout << int(model.write_valid) << ' ' << int(model.read_valid) << ' '
                  << std::hex << encode(model.read_key) << ' ' << encode(model.read_value) << std::dec << std::endl;
    }
    model.final();
}
