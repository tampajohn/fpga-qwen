"""Independent no-logits prefill wire adapter for real RTL validation."""
try:
    from .benchmark_blocks import BlockProtocolModel
except ImportError:
    from benchmark_blocks import BlockProtocolModel


class PrefillProtocolModel(BlockProtocolModel):
    def prefill(self,tokens,cancel=None):
        tokens=list(tokens)
        if not 1<=len(tokens)<=8:raise ValueError('Prefill count must be 1..8')
        if any(type(t) is not int or not 0<=t<151936 for t in tokens):raise ValueError('Invalid prefill token')
        fields=self.command(' '.join(['prefill',str(len(tokens)),*map(str,tokens)]),cancel).split()
        if len(fields)!=3 or fields[:2]!=['prefill',str(len(tokens))]:
            raise RuntimeError(f'Invalid prefill response: {fields}')
        final=int(fields[2])
        if not len(tokens)<=final<=4096:raise RuntimeError('Invalid prefill position')
        return [(0,p,0.0)for p in range(final-len(tokens)+1,final+1)]
