// Bounded projection metadata validation. No weights or activations are read.
#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
extern "C" void validate_weight_span(int64_t index,int64_t count);
struct ProjectionDescriptor { int64_t base; int rows, width, output_offset; };
inline void validate_projection_group(const std::array<ProjectionDescriptor,3>& jobs,
                                     int count,int tokens,const double* x,const double* y) {
#ifdef PREFILL_B8
        constexpr int max_tokens=8;
#else
        constexpr int max_tokens=4;
#endif
        if(count<2 || count>3 || tokens<1 || tokens>max_tokens || !x || !y)
            throw std::runtime_error("Invalid projection group");
        int total=0;
        for(int job=0;job<count;++job) {
            const auto& desc=jobs[job];
            if(desc.rows<1 || desc.rows>3072 || desc.width!=1024 || desc.output_offset!=total ||
               desc.rows>6144-total) throw std::runtime_error("Invalid projection descriptor");
            validate_weight_span(desc.base,int64_t(desc.rows)*desc.width);
            total+=desc.rows;
        }
}
