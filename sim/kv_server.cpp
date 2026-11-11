#include "Vkv_cursor.h"
#include "verilated.h"
#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    Verilated::commandArgs(argc,argv);
    try {
        Vkv_cursor d;
        auto tick=[&]() { d.clk=0;d.eval();d.clk=1;d.eval(); };
        auto reset=[&]() { d.rst_n=0;d.cmd_valid=0;d.rsp_ready=0;tick();
                          d.rst_n=1;tick(); };
        d.cmd_op=0;d.cmd_count=0;reset();
        int op,count;
        while(std::cin>>op>>count) {
            if(op==-1) reset();
            else {
                if(op<0 || op>7 || count<0 || count>65535) throw std::runtime_error("invalid command encoding");
                if(!d.cmd_ready) throw std::runtime_error("cursor not ready");
                d.cmd_op=op;d.cmd_count=count;d.cmd_valid=1;tick();d.cmd_valid=0;
                if(!d.rsp_valid) throw std::runtime_error("missing response");
                std::array<unsigned,6> before={d.rsp_error,d.active,d.committed,d.reserved,d.written,d.visible_end};
                // The requester must be blocked while the response is stalled.
                d.cmd_valid=1;d.cmd_op=4;d.cmd_count=0;
                for(int i=0;i<3;i++) {
                    tick();
                    std::array<unsigned,6> after={d.rsp_error,d.active,d.committed,d.reserved,d.written,d.visible_end};
                    if(d.cmd_ready || !d.rsp_valid || before!=after)
                        throw std::runtime_error("state changed under response backpressure");
                }
                d.cmd_valid=0;d.rsp_ready=1;tick();d.rsp_ready=0;
            }
            std::cout<<unsigned(d.rsp_error)<<' '<<unsigned(d.active)<<' '<<d.committed<<' '
                     <<d.reserved<<' '<<d.written<<' '<<d.visible_end<<std::endl;
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
