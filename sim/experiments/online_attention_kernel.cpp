// Fixture transport and timing only. Neural arithmetic stays in SystemVerilog.
#include "Vonline_attention_kernel.h"
#include <verilated.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
int main(int argc,char**argv) {
    Verilated::commandArgs(argc,argv);
    if(argc!=6) throw std::runtime_error("method count length scenario iterations required");
    int method=std::stoi(argv[1]), count=std::stoi(argv[2]), length=std::stoi(argv[3]), scenario=std::stoi(argv[4]), iterations=std::stoi(argv[5]);
    if(method<0||method>1||count<1||count>8||length<count||length>4096||scenario<0||scenario>2||iterations<1||iterations>100000) throw std::runtime_error("Invalid arguments");
    auto m=std::make_unique<Vonline_attention_kernel>();
    m->method=method;m->length=length;m->count=count;
    const double factor=scenario==1?64.0:scenario==2?1.0/1024.0:1.0;
    for(int j=0;j<128;j++) for(int q=0;q<8;q++) m->queries[j*8+q]=q<count?((q*13+j*7)%19-9)/16.0*factor:0.0;
    for(int t=0;t<4096;t++) for(int j=0;j<128;j++) {
        m->keys[t*128+j]=((t*11+j*3)%23-11)/16.0*factor;
        m->values[t*128+j]=((t*5+j*17)%29-14)/8.0;
    }
    auto tick=[&](){m->clk=0;m->eval();m->clk=1;m->eval();};
    tick();
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<iterations;i++) tick();
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<std::setprecision(17)<<"SECONDS "<<seconds<<"\n";
    for(int q=0;q<count;q++) for(int j=0;j<128;j++) std::cout<<"RESULT "<<q<<' '<<j<<' '<<m->result[j*8+q]<<'\n';
    m->final();
}
