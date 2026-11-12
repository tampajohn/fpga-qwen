"""Bookkeeping only for reusable, complete, profile-specific B8 prefill blocks."""
POLICY = 'fp64_b8_aligned_v1'
FMA_POLICY = 'fp32_fma_b8_aligned_v1'
PROFILE_POLICIES = {
    'fp32_b1_fp64_prefill': POLICY,
    'fp32_b1_fp32_fma_prefill': FMA_POLICY,
}


class CanonicalPrefix:
    def __init__(self):
        self.clear()

    def clear(self):
        self.end = 0
        self.epoch = None
        self.signature = None

    def bind(self, epoch, signature=None):
        if epoch is not self.epoch or signature != self.signature:
            self.clear()
            self.epoch = epoch
            self.signature = signature

    def reuse(self, shared, prefill_length):
        return min(shared, self.end, prefill_length) // 8 * 8

    def truncate(self, position):
        if position < 0 or position % 8:
            raise ValueError('Canonical prefix must end on an eight-token boundary')
        self.end = min(self.end, position)

    def acknowledge(self, start, count):
        if start != self.end or start % 8 or not 1 <= count <= 8:
            raise RuntimeError('Noncanonical prefill progression')
        if count == 8:
            self.end = start + count
