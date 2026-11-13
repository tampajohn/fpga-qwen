"""Source preparation only; copies independent oracle math, adds observational copies."""
from pathlib import Path
import difflib,hashlib,json
ROOT=Path(__file__).resolve().parents[3]
original=ROOT/'tests/test_full_model.py';text=original.read_text();start=text.index('class NumpyQwen:');end=text.index('\n\nclass FullModelTests',start)
base=text[start:end];s=base
s=s.replace('        self.cache = []\n\n    def reset', '        self.cache = []\n        self.trace_rows = {}\n        self.trace_position = 62\n\n    def record(self, name, layer, value):\n        if len(self.cache) == self.trace_position:\n            self.trace_rows[(layer,name)] = value.copy()\n        return value\n\n    def reset',1)
s=s.replace("            h = norm(x, weight('input_layernorm'))", "            h = norm(x, weight('input_layernorm'))\n            self.record('input_norm_h',layer,h)")
s=s.replace("(weight('self_attn.q_proj') @ h).reshape(16, 128)","self.record('q_projection',layer,(weight('self_attn.q_proj') @ h).reshape(16, 128))")
s=s.replace("(weight('self_attn.k_proj') @ h).reshape(8, 128)","self.record('k_projection',layer,(weight('self_attn.k_proj') @ h).reshape(8, 128))")
s=s.replace("            v = (weight('self_attn.v_proj') @ h).reshape(8, 128)", "            v = (weight('self_attn.v_proj') @ h).reshape(8, 128)\n            self.record('v_projection',layer,v)\n            self.record('q_rope',layer,q)\n            self.record('k_rope',layer,k)")
s=s.replace('            keys, vals = np.repeat(keys, 2, axis=1), np.repeat(vals, 2, axis=1)', "            self.record('canonical_keys',layer,keys)\n            self.record('canonical_values',layer,vals)\n            keys, vals = np.repeat(keys, 2, axis=1), np.repeat(vals, 2, axis=1)")
s=s.replace("            x += weight('self_attn.o_proj') @ attended", "            self.record('attended',layer,attended)\n            x += weight('self_attn.o_proj') @ attended\n            self.record('post_attention_residual',layer,x)")
s=s.replace("            h = norm(x, weight('post_attention_layernorm'))", "            h = norm(x, weight('post_attention_layernorm'))\n            self.record('post_norm_h',layer,h)")
s=s.replace("            x += weight('mlp.down_proj') @ activated", "            self.record('activated_mlp',layer,activated)\n            x += weight('mlp.down_proj') @ activated\n            self.record('post_mlp_residual',layer,x)")
preamble='''# Isolated observational copy; original arithmetic retained. No runtime executed by generator.
import os
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.experiment import read_header

'''
out=Path(__file__).parent/'numpy_trace_oracle.py';assert not out.exists();out.write_text(preamble+s+'\n')
(Path(__file__).parent/'numpy_trace_oracle.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='original NumpyQwen class',tofile='observational NumpyQwen class')))
(Path(__file__).parent/'numpy_trace_oracle-provenance.json').write_text(json.dumps(dict(original=str(original),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),diagnostic=str(out),diagnostic_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),target_position=62,source_only=True,arithmetic='Original expressions unchanged except projection values wrapped in identity-returning record(); observational arrays copied, no re-evaluation.'),indent=2)+'\n')
