import unittest
from tools.suffix import SuffixDrafter, greedy_accept


class SuffixTests(unittest.TestCase):
    def test_known_continuation_and_longest_match(self):
        d = SuffixDrafter(min_match=2, max_match=4)
        self.assertEqual(d.propose([1, 2, 3, 4, 5, 1, 2], 3), [3, 4, 5])
        self.assertEqual(d.propose([1, 2, 3, 7, 2, 3, 8, 1, 2, 3], 1), [7])
        self.assertEqual(d.propose([1, 2, 3], 4), [])
        self.assertEqual(d.propose([1, 2, 1, 2], 0), [])

    def test_no_cross_document_continuation(self):
        d = SuffixDrafter(history=[[7, 8], [9, 10, 11]])
        self.assertEqual(d.propose([7, 8], 3), [])
        self.assertEqual(d.propose([9, 10], 3), [11])

    def test_most_recent_occurrence(self):
        d = SuffixDrafter(max_match=2)
        self.assertEqual(d.propose([1, 2, 3, 1, 2, 4, 1, 2], 1), [4])

    def test_rejection_everywhere_and_bonus(self):
        draft = [10, 11, 12, 13]
        for accepted in range(5):
            predictions = draft[:accepted] + [99] * (5-accepted)
            result, count = greedy_accept(draft, predictions)
            self.assertEqual(count, accepted)
            self.assertEqual(result, draft[:accepted] + [99])
        self.assertEqual(greedy_accept([], [7]), ([7], 0))
        with self.assertRaises(ValueError):
            greedy_accept([1], [1])


if __name__ == '__main__':
    unittest.main()
