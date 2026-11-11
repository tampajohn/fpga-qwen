#include "Vq4_matvec.h"
#include "verilated.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

uint32_t read32(std::ifstream& f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4);
    if(!f) throw std::runtime_error("truncated header");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
void write32(std::ofstream& f, uint32_t n) {
    for(int b=0;b<4;b++) f.put(char((n>>(8*b))&255));
}
int main(int argc, char** argv) {
    Verilated::commandArgs(argc,argv);
    try {
        if(argc!=4) throw std::runtime_error("usage: linear weights.bin activations.bin output.bin");
        std::ifstream wf(argv[1],std::ios::binary), xf(argv[2],std::ios::binary);
        if(read32(wf)!=0x34574d51 || read32(xf)!=0x38584d51) throw std::runtime_error("bad magic");
        uint32_t rows=read32(wf), k=read32(wf), tokens=read32(xf), xk=read32(xf);
        if(!rows || rows>3072 || !k || k>3072 || k%16 || !tokens || tokens>8 || xk!=k)
            throw std::runtime_error("unsupported shape");
        std::vector<int8_t> w(rows*k), x(tokens*k);
        wf.read(reinterpret_cast<char*>(w.data()),w.size());
        xf.read(reinterpret_cast<char*>(x.data()),x.size());
        if(!wf || !xf) throw std::runtime_error("truncated payload");
        for(auto n:w) if(n < -8 || n > 7) throw std::runtime_error("weight outside int4");
        Vq4_matvec d;
        uint64_t cycles=0;
        auto tick=[&]() { d.clk=0;d.eval();d.clk=1;d.eval();++cycles; };
        d.rst_n=0;d.x_we=0;d.start=0;d.w_valid=0;d.w_last=0;d.y_ready=0;
        d.x_addr=0;d.x_data=0;d.token_count=0;d.w_data=0;
        tick();tick();d.rst_n=1;tick();cycles=0;
        for(uint32_t i=0;i<k;i++) {
            uint64_t data=0;
            for(uint32_t t=0;t<tokens;t++) data |= uint64_t(uint8_t(x[t*k+i])) << (t*8);
            d.x_we=1;d.x_addr=i;d.x_data=data;tick();
        }
        d.x_we=0;
        std::vector<int32_t> y(tokens*rows);
        for(uint32_t r=0;r<rows;r++) {
            d.start=1;d.token_count=tokens;tick();d.start=0;
            for(uint32_t i=0;i<k;i+=16) {
                uint64_t data=0;
                for(int l=0;l<16;l++) data |= uint64_t(uint8_t(w[r*k+i+l])&15) << (l*4);
                if(!d.w_ready) throw std::runtime_error("weight stream not ready");
                d.w_data=data;d.w_valid=1;d.w_last=(i+16==k);tick();
            }
            d.w_valid=0;d.w_last=0;
            if(!d.y_valid) throw std::runtime_error("missing result");
            for(uint32_t t=0;t<tokens;t++) y[t*rows+r]=int32_t(d.y_data[t]);
            d.y_ready=1;tick();d.y_ready=0;
        }
        std::ofstream out(argv[3],std::ios::binary);
        write32(out,0x32594d51);write32(out,tokens);write32(out,rows);
        for(auto n:y) write32(out,uint32_t(n));
        out.close();if(!out) throw std::runtime_error("output write failed");
        std::cout<<"{\"cycles\":"<<cycles<<",\"weight_bytes\":"<<uint64_t(rows)*k/2
                 <<",\"activation_bytes\":"<<uint64_t(tokens)*k
                 <<",\"output_bytes\":"<<uint64_t(tokens)*rows*4<<"}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
