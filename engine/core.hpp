#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace qwen_cpu {
inline constexpr int vocabulary = 151936;
inline constexpr int prefix_reuse_chunk = 512;
struct Options {
    int decode_threads = 12;
    int prefill_threads = 6;
    int context = 4096;
};
// One caller at a time. FP16 stored matrices, FP32 norms/math, FP16 KV.
// All input IDs and total context capacity must be checked before mutation.
class Model {
public:
    Model(const std::string& weights_file, Options options);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    void reset();
    // Retain KV [0, position), discard logits. Invalid positions leave state
    // unchanged. Exact prefix reuse additionally requires canonical batch bounds.
    void truncate(int position);
    // Append IDs, computing final-position full-vocabulary logits.
    void prefill(const std::vector<int>& ids);
    void step(int token);
    const std::vector<float>& logits() const;
    int position() const;
    // Auxiliary decode encoding, populated only for exactly representable matrices.
    std::size_t lossless_bf16_cache_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
