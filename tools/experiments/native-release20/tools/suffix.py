"""Simple longest-match n-gram drafter. No neural draft model or future tokens."""


class SuffixDrafter:
    def __init__(self, min_match=2, max_match=12, max_context=4096, history=()):
        if not 1 <= min_match <= max_match or max_context < max_match:
            raise ValueError('Invalid matching limits')
        self.min_match = min_match
        self.max_match = max_match
        self.max_context = max_context
        self.history = tuple(tuple(seq) for seq in history)

    def propose(self, context, count):
        if count < 0:
            raise ValueError('Negative proposal length')
        if count == 0:
            return []
        context = tuple(context[-self.max_context:])
        # Search the current context first, then history from newest to oldest.
        sources = (context,) + tuple(reversed(self.history))
        for n in range(min(self.max_match, len(context)), self.min_match-1, -1):
            suffix = context[-n:]
            for seq in sources:
                # The matching occurrence must have at least one known next
                # token. A current-context suffix cannot match itself here.
                for start in range(len(seq)-n-1, -1, -1):
                    if seq[start:start+n] == suffix:
                        return list(seq[start+n:start+n+count])
        return []


def greedy_accept(draft, predictions):
    """Accept the matching prefix, then emit the target correction/bonus token."""
    if len(predictions) != len(draft)+1:
        raise ValueError('Need one target prediction per draft token plus a bonus')
    accepted = 0
    for proposed, predicted in zip(draft, predictions):
        if proposed != predicted:
            break
        accepted += 1
    return list(draft[:accepted]) + [predictions[accepted]], accepted
