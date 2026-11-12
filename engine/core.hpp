#pragma once
#include <cstddef>
#include <memory>
#include <span>
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
struct VerificationRows {
    // Borrowed row-major [2, vocabulary] and [2, 1024] diagnostic/target outputs.
    // Valid until commit/reset/truncate/another successful model mutation.
    std::span<const float> logits;
    std::span<const float> hidden;
    int base_position;
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
    // Speculatively evaluate two consumed IDs using exact B1 arithmetic. Public
    // position/logits/hidden remain at the base while results are pending.
    // Pending work blocks prefill/step/verify until commit, reset or truncate.
    // Validation failures preserve state; execution failures reset logical state.
    VerificationRows verify_two(int first, int second);
    // Retain one/two verified inputs, restore its current logits/hidden, and
    // invalidate both borrowed verification rows. Does not sample or touch RNG.
    void commit_verified(int count);
    const std::vector<float>& logits() const;
    // Diagnostic view of the final normalized hidden state backing current
    // logits; invalid after reset/rewind or any subsequent inference mutation.
    std::span<const float> hidden() const;
    int position() const;
    // Auxiliary decode encoding, populated only for exactly representable matrices.
    std::size_t lossless_bf16_cache_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
