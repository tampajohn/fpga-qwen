#include "Vq4_matvec.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct Fixture { std::string name; uint32_t rows, k; std::vector<int8_t> w; };
struct Stats { uint64_t cycles=0, beats=0, rows=0, load_cycles=0, wait_cycles=0; };

class Bench {
    Vq4_matvec d;
public:
    Stats s;
    void tick() { d.clk=0; d.eval(); d.clk=1; d.eval(); ++s.cycles; }
    Bench() {
        d.rst_n=0; d.x_we=0; d.start=0; d.w_valid=0; d.w_last=0;
        d.y_ready=0; d.x_addr=0; d.x_data=0; d.token_count=0; d.w_data=0;
        tick(); tick(); d.rst_n=1; tick(); s={};
    }
    Stats run(const Fixture& f, int width, int gap, int latency, uint32_t seed) {
        if (f.k == 0 || f.k > 3072 || f.k % 16) throw std::runtime_error("unsupported K");
        std::mt19937 rng(seed);
        std::vector<std::vector<int8_t>> x(8, std::vector<int8_t>(f.k));
        for(auto& v:x) for(auto& a:v) a=int8_t(int(rng()%256)-128);
        // Always exercise both signed extremes, independent of the random seed.
        x[0][0]=-128; x[0][1]=127;
        const auto before=s;
        for(int first=0;first<8;first+=width) {
            for(uint32_t k=0;k<f.k;k++) {
                uint64_t packed=0;
                for(int t=0;t<width;t++) packed |= uint64_t(uint8_t(x[first+t][k])) << (8*t);
                d.x_we=1; d.x_addr=k; d.x_data=packed; tick(); ++s.load_cycles;
            }
            d.x_we=0;
            for(uint32_t r=0;r<f.rows;r++) {
                d.start=1; d.token_count=width; tick(); d.start=0;
                if(!d.w_ready) throw std::runtime_error("not ready after start");
                // Abstract external memory source: startup delay per row followed
                // by one 64-bit beat every (gap+1) clocks. Not a DDR/HBM controller.
                for(int i=0;i<latency;i++) { tick(); ++s.wait_cycles; }
                for(uint32_t k=0;k<f.k;k+=16) {
                    uint64_t packed=0;
                    for(int l=0;l<16;l++) packed |= uint64_t(uint8_t(f.w[r*f.k+k+l])&15) << (4*l);
                    d.w_data=packed; d.w_last=(k+16==f.k); d.w_valid=1;
                    if(!d.w_ready) throw std::runtime_error("weight backpressure unexpected");
                    tick(); ++s.beats; d.w_valid=0;
                    if(k+16<f.k) for(int i=0;i<gap;i++) { tick(); ++s.wait_cycles; }
                }
                d.w_last=0;
                if(!d.y_valid) throw std::runtime_error("missing result");
                for(int t=0;t<8;t++) {
                    int32_t expected=0;
                    if(t<width) for(uint32_t k=0;k<f.k;k++)
                        expected += int32_t(f.w[r*f.k+k])*int32_t(x[first+t][k]);
                    if(int32_t(d.y_data[t]) != expected)
                        throw std::runtime_error("integer mismatch: "+f.name+" row "+std::to_string(r));
                }
                // Hold sink stalled and ensure the valid result stays unchanged.
                std::array<uint32_t,8> held;
                for(int t=0;t<8;t++) held[t]=d.y_data[t];
                for(int n=0;n<3;n++) {
                    tick();
                    if(!d.y_valid || d.w_ready) throw std::runtime_error("invalid stalled handshake");
                    for(int t=0;t<8;t++) if(d.y_data[t]!=held[t]) throw std::runtime_error("unstable output");
                }
                d.y_ready=1; tick(); d.y_ready=0;
                if(d.y_valid) throw std::runtime_error("result was not consumed");
                s.rows+=width;
            }
        }
        return {s.cycles-before.cycles,s.beats-before.beats,s.rows-before.rows,
                s.load_cycles-before.load_cycles,s.wait_cycles-before.wait_cycles};
    }
    void reset_mid_row() {
        d.start=1; d.token_count=8; tick(); d.start=0;
        d.rst_n=0; tick(); d.rst_n=1; tick();
        if(d.y_valid || d.w_ready) throw std::runtime_error("reset failed");
    }
};

uint32_t u32(std::ifstream& f) {
    uint8_t b[4]; f.read(reinterpret_cast<char*>(b),4);
    if(!f) throw std::runtime_error("truncated fixture header");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
int main(int argc,char** argv) {
    Verilated::commandArgs(argc,argv);
    try {
        if(argc!=2) throw std::runtime_error("usage: Vq4_matvec fixture-directory");
        std::vector<Fixture> fixtures;
        for(const auto& name:{"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}) {
            std::ifstream f(std::string(argv[1])+"/"+name+".bin",std::ios::binary);
            if(!f) throw std::runtime_error("missing fixture "+std::string(name));
            if(u32(f)!=0x34574d51) throw std::runtime_error("bad magic");
            Fixture a; a.name=name; a.rows=u32(f); a.k=u32(f);
            if(a.rows==0 || a.rows>64 || a.k==0 || a.k>3072 || a.k%16)
                throw std::runtime_error("invalid fixture dimensions");
            a.w.resize(a.rows*a.k); f.read(reinterpret_cast<char*>(a.w.data()),a.w.size());
            if(!f) throw std::runtime_error("truncated weights");
            for(auto w:a.w) if(w < -8 || w > 7) throw std::runtime_error("weight outside int4");
            fixtures.push_back(a);
        }
        Fixture edge{"signed_edges",4,32,{}};
        for(uint32_t i=0;i<edge.rows*edge.k;i++) edge.w.push_back(int8_t(int(i%16)-8));
        fixtures.push_back(edge);
        Bench b; b.reset_mid_row();
        std::cout<<"matrix,k,rows,seed,token_width,gap_cycles,row_latency,cycles,weight_bytes,outputs,activation_load_cycles,memory_wait_cycles\n";
        for(const auto& f:fixtures) for(uint32_t seed:{17,93})
            for(int gap:{0,3}) for(int width:{1,2,4,8}) {
                auto s=b.run(f,width,gap,12,seed);
                std::cout<<f.name<<','<<f.k<<','<<f.rows<<','<<seed<<','<<width<<','<<gap<<",12,"
                         <<s.cycles<<','<<s.beats*8<<','<<s.rows<<','<<s.load_cycles<<','<<s.wait_cycles<<'\n';
            }
        std::cerr<<"PASS: all RTL outputs match independent signed integer reference; stalls and reset checked.\n";
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
