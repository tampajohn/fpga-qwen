#pragma once
#include <cstdlib>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <string>
namespace query_value {
// Trusted coordinator configuration. Environment supplies the initial policy once.
// This is scheduling metadata only; the actual branch and arithmetic are SV.
struct Policy {
    std::string selector;
    int min_context=-1;
    explicit Policy(const char* value):Policy(std::string(value?value:"OFF")) {}
    explicit Policy(std::string value):selector(std::move(value)) {
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
// Only the idle command coordinator accesses this state. Workers receive an
// integer snapshot through the pool's existing synchronized job publication.
struct State {
    Policy policy{std::getenv("QWEN_EXPERIMENTAL_QUERY_VALUE_PAIRS")};
    uint64_t epoch=0;
};
inline State& coordinator_state() {
    static State state;
    return state;
}
inline const Policy& configured_policy() {return coordinator_state().policy;}
inline uint64_t policy_epoch() {return coordinator_state().epoch;}
inline void set_policy(const std::string& selector) {
    Policy candidate(selector); // Validate fully before mutating live state.
    auto& state=coordinator_state();
    state.policy=std::move(candidate);
    ++state.epoch;
}
}
