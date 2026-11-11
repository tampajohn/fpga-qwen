// Transport/timing only. All quantization, dots, rescaling and error metrics are SV.
#include "Vw8a8_probe.h"
#include "verilated.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

static uint32_t read32(std::ifstream& file) {
    uint32_t value=0;file.read(reinterpret_cast<char*>(&value),4);
    if(!file)throw std::runtime_error("Truncated fixture");return value;
}
static std::vector<uint16_t> raw_weights;
static double expand(uint16_t raw) {
    uint32_t bits=uint32_t(raw)<<16;float value;std::memcpy(&value,&bits,4);return value;
}
static uint16_t read16(std::ifstream& file) {
    uint16_t raw=0;file.read(reinterpret_cast<char*>(&raw),2);
    if(!file)throw std::runtime_error("Truncated BF16 fixture");return raw;
}
static double bf16(std::ifstream& file) {return expand(read16(file));}
extern "C" double weight_value_unchecked(long long index) {return expand(raw_weights[index]);}
extern "C" double attention_key_value(int) {std::abort();}
extern "C" double attention_value_value(int) {std::abort();}
int main(int argc,char** argv) {
    try {
        if(argc<3 || argc>4)throw std::runtime_error("usage: probe fixture check|bench [repetitions]");
        bool bench=std::string(argv[2])=="bench";
        if(!bench && std::string(argv[2])!="check")throw std::runtime_error("Invalid mode");
        int repetitions=argc==4?std::stoi(argv[3]):100;
        if(repetitions<1 || repetitions>100000)throw std::runtime_error("Invalid repetitions");
        std::ifstream file(argv[1],std::ios::binary);
        if(read32(file)!=0x38513857)throw std::runtime_error("Invalid fixture magic");
        uint32_t rows=read32(file),width=read32(file),tokens=read32(file),fixture=read32(file);
        if(rows<1 || rows>16 || width<1 || width>3072 || tokens<1 || tokens>4 || fixture>2)
            throw std::runtime_error("Invalid fixture dimensions");
        auto model=std::make_unique<Vw8a8_probe>();
        model->rows=rows;model->width=width;model->tokens=tokens;model->fixture=fixture;
        for(uint32_t r=0;r<rows;r++)for(uint32_t k=0;k<width;k++) {
            uint16_t raw=read16(file);raw_weights.push_back(raw);model->weights[r*3072+k]=expand(raw);
        }
        if(fixture==1) {
            if(width!=1024)throw std::runtime_error("Embedding fixture width");
            for(int k=0;k<1024;k++)model->norm_weight[k]=bf16(file);
            for(uint32_t t=0;t<tokens;t++)for(int k=0;k<1024;k++)model->embedding[t*1024+k]=bf16(file);
        }
        auto run=[&](int operation) {
            model->operation=operation;model->start=1;
            model->clk=0;model->eval();model->clk=1;model->eval();
            model->start=0;
        };
        auto measure=[&](int operation,int n) {
            auto before=std::chrono::steady_clock::now();
            for(int i=0;i<n;++i)run(operation);
            return std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count()/n;
        };
        run(7);if(model->failures)throw std::runtime_error("Known-answer quantizer test failed");
        run(0);
        double prepare=measure(1,1);
        run(2);run(4);
        if(model->failures)throw std::runtime_error("Independent SV quantization/int64 oracle mismatch");
        run(5);run(9);run(10);
        if(model->failures)throw std::runtime_error("Deployed FP64 worker mismatch");
        double error=model->max_error,rmse=model->relative_rmse;
        std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"width\":"<<width
                 <<",\"tokens\":"<<tokens<<",\"fixture\":"<<fixture<<",\"exact_integer_oracle\":true"
                 <<",\"weight_prepare_seconds\":"<<prepare<<",\"max_error\":"<<error
                 <<",\"relative_rmse\":"<<rmse;
        if(bench) {
            run(2);run(9);
            double total=measure(2,repetitions),quant=measure(6,repetitions);
            double dot=measure(3,repetitions),floating=measure(9,repetitions);
            std::cout<<",\"repetitions\":"<<repetitions<<",\"quantize_dot_rescale_seconds\":"<<total
                     <<",\"activation_quantize_seconds\":"<<quant<<",\"dot_rescale_seconds\":"<<dot
                     <<",\"deployed_fp64_projection_seconds\":"<<floating;
        }
        std::cout<<"}\n";model->final();
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
