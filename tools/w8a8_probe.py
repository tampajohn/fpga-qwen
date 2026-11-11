"""Isolated SV quantization probe: host copies BF16 bytes and invokes tools only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from tools.experiment import locate_model, read_header
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'build/w8a8'

def fixtures():
    checkpoint = locate_model(None) / 'model.safetensors'
    header, start, _ = read_header(checkpoint)
    DEST.mkdir(parents=True, exist_ok=True)
    def raw(name, row, width):
        spec = header[name]
        assert spec['dtype'] == 'BF16'
        stride = spec['shape'][-1]
        with checkpoint.open('rb') as f:
            f.seek(start + spec['data_offsets'][0] + row * stride * 2)
            data = f.read(width * 2)
        assert len(data) == width * 2
        return data
    cases = []
    for width in [7, 8, 63, 64, 65, 1024, 3072]:
        for tokens in [1, 4]:
            cases.append((f'synthetic-{width}-b{tokens}',16,width,tokens,0))
    cases += [('q-real-b1',16,1024,1,1), ('q-real-b4',16,1024,4,1), ('zero',4,65,4,2)]
    manifest=[]
    for name, rows, width, tokens, kind in cases:
        matrix='model.layers.0.' + ('mlp.down_proj.weight' if width>1024 else 'self_attn.q_proj.weight')
        payload = struct.pack('<IIIII',0x38513857,rows,width,tokens,kind)
        payload += b''.join(raw(matrix,r,width) for r in range(rows)) if kind!=2 else bytes(rows*width*2)
        if kind==1:
            payload += raw('model.layers.0.input_layernorm.weight',0,1024)
            for token in [9707,11,1917,0][:tokens]:
                payload += raw('model.embed_tokens.weight',token,1024)
        path=DEST/(name+'.bin');path.write_bytes(payload)
        manifest.append(dict(name=name, matrix=matrix, rows=rows,width=width,tokens=tokens,fixture=kind,sha256=hashlib.sha256(payload).hexdigest()))
    (DEST/'fixtures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','build','check','bench']);args=parser.parse_args()
    if args.action=='prepare': fixtures()
    elif args.action=='build':
        DEST.mkdir(parents=True,exist_ok=True)
        subprocess.run(['verilator','--cc','--exe','--build','-j','4','--top-module','w8a8_probe','--Mdir',str(DEST/'obj'),'-Wall','-Wno-BLKSEQ','-O3','-DUNCHECKED_WEIGHTS','-DMASK_ACTIVATION_INDEX','-DUNSIGNED_PROJECTION_INDEX','-DPARALLEL_ATTENTION','-DBATCH_PREFILL','-DBLOCK_UNROLL','-CFLAGS','-std=c++17 -O3 -mcpu=native -flto=thin','-LDFLAGS','-flto=thin','-MAKEFLAGS','OPT_FAST=-O3',str(ROOT/'rtl/w8a8_probe.sv'),str(ROOT/'rtl/projection_worker.sv'),str(ROOT/'sim/w8a8_probe.cpp')],check=True)
    elif args.action=='bench':
        results=[]
        for name in ['q-real-b1','q-real-b4','synthetic-3072-b1','synthetic-3072-b4']:
            for repeat in range(3):
                output=subprocess.check_output([str(DEST/'obj/Vw8a8_probe'),str(DEST/(name+'.bin')),'bench','1000'],text=True)
                results.append(dict(case=name,repeat=repeat,**json.loads(output)))
        (DEST/'timing-deployed.json').write_text(json.dumps(results,indent=2)+'\n')
    else:
        manifest=fixtures();results=[]
        for case in manifest:
            output=subprocess.check_output([str(DEST/'obj/Vw8a8_probe'),str(DEST/(case['name']+'.bin')),'check'],text=True)
            results.append(dict(case=case['name'],**json.loads(output)))
        report=dict(binary_sha256=hashlib.sha256((DEST/'obj/Vw8a8_probe').read_bytes()).hexdigest(),results=results)
        (DEST/'correctness.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
