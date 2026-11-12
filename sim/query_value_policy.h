#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace query_value {
// Trusted process configuration, captured once and shared by every pool.
// This is scheduling metadata only; the actual branch and arithmetic are SV.
struct Policy {
    std::string selector;
    int min_context=-1;
    explicit Policy(const char* value):selector(value?value:"OFF") {
        if(selector=="OFF") return;
        if(selector=="ON") {min_context=0;return;}
        const std::string prefix="MIN_CONTEXT:";
        if(selector.compare(0,prefix.size(),prefix)==0) {
            const std::string digits=selector.substr(prefix.size());
            if(!digits.empty() && digits.size()<=4 && digits.find_first_not_of("0123456789")==std::string::npos) {
                const int n=std::stoi(digits);
                if(n>=1 && n<=4096) {min_context=n;return;}
            }
        }
        throw std::runtime_error("QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS must be OFF, ON or MIN_CONTEXT:N (1..4096)");
    }
};
inline const Policy& configured_policy() {
    static const Policy policy(std::getenv("QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS"));
    return policy;
}
}
