"""Source-only isolated instrumentation; never compiles or runs inference."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[3]
source=root/'engine'
out=root/'build/native-cpu/build18-phase-diagnostic-source'
manifest=json.loads((root/'build/native-cpu/build-18/manifest.json').read_text())
for name in ('core.cpp','core.hpp','main.cpp'):
    p=(source/name).resolve()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['source_sha256'][str(p)],name
out.mkdir(exist_ok=False)
s=(source/'core.cpp').read_text()
a=s.index('    void chunk(');b=s.index('    void verify_chunk(',a)
chunk=s[a:b]
def replace(old,new):
    global chunk
    assert chunk.count(old)==1,(old,chunk.count(old))
    chunk=chunk.replace(old,new)
replace('        x.resize(size_t(batch)*H);', '        auto& dr=diag_chunks.at(diag_count++);dr.base=pos;dr.batch=batch;\n        auto dt=std::chrono::steady_clock::now();\n        auto mark=[&](int layer,int phase){auto now=std::chrono::steady_clock::now();dr.ns[layer][phase]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-dt).count();dt=now;};\n        x.resize(size_t(batch)*H);')
replace('        for(int layer=0;layer<L;++layer)', '        mark(28,0);\n        for(int layer=0;layer<L;++layer)')
replace('            if(batch==1) {\n                project_group<3>', '            mark(layer,1);\n            if(batch==1) {\n                project_group<3>')
replace('            const float* qn=', '            mark(layer,2);\n            const float* qn=')
replace('            if(batch>1) {\n                // One item', '            mark(layer,3);\n            if(batch>1) {\n                // One item')
replace('            project(tensor(p+"self_attn.o_proj.weight")', '            mark(layer,4);\n            project(tensor(p+"self_attn.o_proj.weight")')
replace('            nw=tensor(p+"post_attention_layernorm.weight")', '            mark(layer,5);\n            nw=tensor(p+"post_attention_layernorm.weight")')
replace('            if(batch==1) {\n                project_group<2>', '            mark(layer,6);\n            if(batch==1) {\n                project_group<2>')
replace('            // vForce is', '            mark(layer,7);\n            // vForce is')
replace('            project(tensor(p+"mlp.down_proj.weight")', '            mark(layer,8);\n            project(tensor(p+"mlp.down_proj.weight")')
replace('            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];\n        }', '            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];\n            mark(layer,9);\n        }')
replace('tensor("model.norm.weight").norm,last_hidden.data(),H);', 'tensor("model.norm.weight").norm,last_hidden.data(),H);\n        mark(28,10);')
fields='''    // Diagnostic coordinator-only wall boundaries; no worker-clock reads.
    struct DiagChunk {int base=0,batch=0;long long ns[29][11]{};};
    std::array<DiagChunk,8> diag_chunks{};
    size_t diag_count=0;
    void diag_emit(int base,int count,bool prefill,long long total,long long head) {
        std::ostringstream os;
        os << "{\\"diagnostic\\":\\"prefill_phase_wall_v1\\",\\"base\\":" << base
           << ",\\"tokens\\":" << count << ",\\"prefill\\":" << (prefill?"true":"false")
           << ",\\"append_ns\\":" << total << ",\\"head_ns\\":" << head << ",\\"chunks\\":[";
        for(size_t c=0;c<diag_count;++c){if(c)os<<',';const auto& d=diag_chunks[c];
            os<<"{\\"base\\":"<<d.base<<",\\"batch\\":"<<d.batch<<",\\"layer_phase_ns\\":[";
            for(int l=0;l<29;++l){if(l)os<<',';os<<'[';for(int p=0;p<11;++p){if(p)os<<',';os<<d.ns[l][p];}os<<']';}os<<"]}";
        }
        os<<"]}\\n";std::cerr<<os.str();
    }
'''
s=s[:a]+fields+chunk+s[b:]
s=s.replace('#include <fstream>','#include <fstream>\n#include <iostream>\n#include <sstream>')
needle='        Pool& pool=prefill?*prefill_pool:*decode_pool;'
assert s.count(needle)==1
s=s.replace(needle,'        diag_chunks={};diag_count=0;const int diag_base=pos;\n        auto diag_start=std::chrono::steady_clock::now();\n'+needle)
needle='        output.resize(vocabulary);project(tensor("model.embed_tokens.weight"),last_hidden.data(),output.data(),1,pool);'
assert s.count(needle)==1
s=s.replace(needle,'        auto diag_head=std::chrono::steady_clock::now();\n'+needle+'\n        auto diag_end=std::chrono::steady_clock::now();\n        diag_emit(diag_base,int(ids.size()),prefill,std::chrono::duration_cast<std::chrono::nanoseconds>(diag_end-diag_start).count(),std::chrono::duration_cast<std::chrono::nanoseconds>(diag_end-diag_head).count());')
(out/'core.cpp').write_text(s)
for name in ('core.hpp','main.cpp'):shutil.copyfile(source/name,out/name)
meta=dict(diagnostic_only=True,base_build='build-18',base_source_sha256=manifest['source_sha256'],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},phases=['setup_embedding_ranges','input_norm','qkv_projection','qk_norm_rope_kv_store','attention','output_projection_residual','post_attention_norm','gate_up_projection','activation','down_projection_residual','final_norm'],layer_rows='0..27 actual layers; row28 chunk setup/final norm',clock='steady_clock nanoseconds, coordinator synchronous boundaries',limitations=['Instrumented wall attribution only; not an uninstrumented performance benchmark.','Includes worker scheduling, synchronization and contention in enclosing phase.','Timestamp/storage overhead is charged to next phase; formatting and stderr I/O occur after append_ns.','No CPU-utilization or sleeping-thread attribution; no subphase attribution within attention.','At most eight512-token chunks per4096 context.','Require full-logit identity against immutable18 captures before interpreting diagnostics.'])
(out/'instrumentation.json').write_text(json.dumps(meta,indent=2)+'\n')
print(out)
