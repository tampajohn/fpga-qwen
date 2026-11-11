import random
import unittest
from tools.kv import RTLCursor


class CursorTests(unittest.TestCase):
    def test_visibility_rejection_and_overwrite(self):
        with RTLCursor() as c:
            self.assertEqual(c.command(0, 4).visible_end, 0)
            self.assertEqual(c.command(1, 2).visible_end, 2)
            self.assertEqual(c.command(2, 2, allow_error=True).error, 1)
            c.command(1, 2)
            self.assertEqual(c.command(2, 3).committed, 3)
            self.assertEqual(c.state.visible_end, 3)
            c.command(0, 4)
            c.command(1, 4)
            self.assertEqual(c.state.visible_end, 7)
            self.assertEqual(c.command(2, 1).committed, 4)
            c.command(0, 1)
            self.assertEqual(c.state.committed, 4)  # correction overwrites rejected slot
            c.command(1, 1)
            self.assertEqual(c.command(2, 1).committed, 5)
            self.assertEqual(c.command(4, 2).visible_end, 2)  # exact prefix reuse

    def test_capacity_zero_commit_abort_and_reset(self):
        with RTLCursor() as c:
            for _ in range(512):
                c.command(0, 8)
                c.command(1, 8)
                c.command(2, 8)
            self.assertEqual(c.state.committed, 4096)
            self.assertEqual(c.command(0, 1, allow_error=True).error, 1)
            self.assertEqual(c.state.committed, 4096)
            c.command(4, 4094)
            self.assertEqual(c.command(0, 3, allow_error=True).error, 1)
            c.command(0, 2)
            c.command(1, 2)
            self.assertEqual(c.command(2, 0).committed, 4094)
            c.command(0, 1)
            c.command(1, 1)
            self.assertEqual(c.command(3).visible_end, 4094)
            c.command(0, 2)
            c.command(1, 1)
            self.assertEqual(c.command(-1).visible_end, 0)
            self.assertEqual(c.state.active, 0)

    def test_randomized_command_sequences(self):
        # Independent behavioral model stores a list of committed token slots
        # and a tentative write bitmap instead of arithmetic cursor registers.
        rng = random.Random(73)
        committed, tentative = [], None
        with RTLCursor() as c:
            for _ in range(2000):
                op, count = rng.randrange(-1, 8), rng.choice([0, 1, 2, 4, 8, 9, 4096, 65535])
                error = 0
                if op == -1:
                    committed, tentative = [], None
                elif op == 0 and tentative is None and 1 <= count <= 8 and len(committed)+count <= 4096:
                    tentative = [False] * count
                elif op == 1 and tentative is not None and 0 < count <= tentative.count(False):
                    first = tentative.index(False)
                    tentative[first:first+count] = [True] * count
                elif op == 2 and tentative is not None and all(tentative) and count <= len(tentative):
                    committed.extend(tentative[:count])
                    tentative = None
                elif op == 3 and tentative is not None and count == 0:
                    tentative = None
                elif op == 4 and tentative is None and count <= len(committed):
                    del committed[count:]
                else:
                    error = 1
                state = c.command(op, count, allow_error=True)
                self.assertEqual(state.error, error)
                self.assertEqual(state.active, int(tentative is not None))
                self.assertEqual(state.committed, len(committed))
                self.assertEqual(state.reserved, len(tentative or []))
                self.assertEqual(state.written, sum(tentative or []))
                self.assertEqual(state.visible_end, len(committed)+sum(tentative or []))


if __name__ == '__main__':
    unittest.main()
