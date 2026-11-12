#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace phase_pool {
enum class Purpose { scalar=0,prefill=1,verification=2 };
struct Policy {
    bool enabled=false;
    Purpose purpose=Purpose::scalar;
    std::array<std::array<uint64_t,3>,2> dispatches{};
    explicit Policy(const char* selector) {
        if(!selector || !*selector || std::strcmp(selector,"OFF")==0)return;
        if(std::strcmp(selector,"PREFILL_12")==0){enabled=true;return;}
        throw std::runtime_error("QWEN_EXPERIMENTAL_PHASE_POOLS must be OFF or PREFILL_12");
    }
    bool prefill_pool()const{return enabled && purpose==Purpose::prefill;}
    void record(){++dispatches[prefill_pool()?1:0][int(purpose)];}
    void validate_config(int workers,int lanes)const {
        if(workers!=8 || lanes!=8)throw std::runtime_error("Phase-pool experiment requires fixed workers8 and lanes8");
    }
    struct Scope {
        Policy& policy;Purpose previous;
        Scope(Policy& p,Purpose next):policy(p),previous(p.purpose){policy.purpose=next;}
        ~Scope(){policy.purpose=previous;}
    };
};
}
