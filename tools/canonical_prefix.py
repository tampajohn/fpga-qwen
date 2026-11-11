"""Bookkeeping only for reusable, complete FP64 B8 prefill blocks."""
POLICY = 'fp64_b8_aligned_v1'


class CanonicalPrefix:
    def __init__(self):
        self.clear()

    def clear(self):
        self.end = 0
        self.epoch = None

    def bind(self, epoch):
        if epoch is not self.epoch:
            self.clear()
            self.epoch = epoch

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
