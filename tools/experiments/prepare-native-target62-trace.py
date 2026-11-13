"""Isolated source preparation only: native build19 numerical phase tracing."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2];base=root/'build/native-cpu';source=root/'engine';dest=base/'native-target62-trace-v1-source';dest.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('core.cpp','core.hpp','main.cpp'):
 assert sha(source/name)==sha(base/'kv-head-major-time32-source'/name),'Expected unchangedbuild19 source'
 text=(source/name).read_text()
 if name=='core.cpp':
  text=text.replace('#include <fstream>\n','#include <fstream>\n#include <cstdlib>\n#include <filesystem>\n#include <iomanip>\n#include <initializer_list>\n#include <sstream>\n',1)
  marker='struct Tensor {int rows=0,cols=0;const Half* half=nullptr;const float* norm=nullptr;std::vector<float> wide;};\n'
  helper=r'''
// Diagnostic-only observer. No inference operands are modified. Enable only
// for root-owned numerical captures; trace I/O invalidates all timing claims.
std::atomic<unsigned long long> diagnostic_capture_sequence{0};
struct Target62Trace {
    static constexpr int target=62;
    bool enabled=false;
    int row=0;
    std::filesystem::path directory;
    std::ofstream manifest;
    static std::string number(unsigned long long n,int width) {
        std::ostringstream out;out<<std::setw(width)<<std::setfill('0')<<n;return out.str();
    }
    Target62Trace(int base,int batch) {
        if(target<base || target>=base+batch)return;
        const char* path=std::getenv("QWEN_NATIVE_TRACE_DIR");
        if(!path || !*path)return;
        const std::filesystem::path root(path);
        std::filesystem::create_directories(root);
        require(std::filesystem::is_directory(root),"Trace root is not a directory");
        directory=root/("capture-"+number(diagnostic_capture_sequence.fetch_add(1),6));
        require(std::filesystem::create_directory(directory),"Trace capture already exists");
        manifest.open(directory/"manifest.jsonl",std::ios::out|std::ios::trunc);
        require(bool(manifest),"Cannot create trace manifest");
        row=target-base;enabled=true;
        manifest<<"{\"kind\":\"start\",\"target_position\":"<<target
                <<",\"base_position\":"<<base<<",\"batch\":"<<batch
                <<",\"target_row\":"<<row<<",\"dtype\":\"float32\",\"byte_order\":\"little\"}\n";
        manifest.flush();require(bool(manifest),"Trace header write failed");
    }
    void write(int layer,const char* phase,const float* data,std::initializer_list<int> shape) {
        if(!enabled)return;
        size_t count=1;for(int dimension:shape){require(dimension>0,"Invalid trace shape");count*=size_t(dimension);}
        const std::string file="layer-"+number(layer,2)+"-"+phase+".f32";
        require(!std::filesystem::exists(directory/file),"Trace tensor already exists");
        std::ofstream output(directory/file,std::ios::out|std::ios::binary|std::ios::trunc);
        require(bool(output),"Cannot create trace tensor");
        output.write(reinterpret_cast<const char*>(data),count*sizeof(float));output.close();
        require(bool(output),"Trace tensor write failed");
        manifest<<"{\"kind\":\"tensor\",\"layer\":"<<layer<<",\"phase\":\""<<phase
                <<"\",\"file\":\""<<file<<"\",\"shape\":[";
        bool first=true;for(int dimension:shape){if(!first)manifest<<",";first=false;manifest<<dimension;}
        manifest<<"],\"elements\":"<<count<<",\"bytes\":"<<count*sizeof(float)
                <<",\"dtype\":\"float32\",\"byte_order\":\"little\"}\n";
        manifest.flush();require(bool(manifest),"Trace manifest write failed");
    }
    void tensor(int layer,const char* phase,const float* data,int stride,std::initializer_list<int> shape) {
        if(enabled)write(layer,phase,data+size_t(row)*stride,shape);
    }
    void finish() {
        if(enabled){manifest<<"{\"kind\":\"complete\",\"completed\":true}\n";manifest.flush();require(bool(manifest),"Trace completion write failed");}
    }
};
'''
  assert text.count(marker)==1;text=text.replace(marker,marker+helper,1)
  start=text.index('    void chunk(const int* ids,int batch,Pool& pool) {');end=text.index('    void verify_chunk(',start)
  chunk=text[start:end]
  def after(old,new):
   global chunk
   assert chunk.count(old)==1,old
   chunk=chunk.replace(old,old+new,1)
  after('        half_gate.resize(gate.size());tanh_gate.resize(gate.size());\n','        Target62Trace trace(pos,batch);\n')
  after('            for(int b=0;b<batch;++b)norm(x.data()+b*H,nw,h.data()+b*H,H);\n','            trace.tensor(layer,"input_norm_h",h.data(),H,{H});\n')
  after('                project(tensor(p+"self_attn.v_proj.weight"),h.data(),v.data(),batch,pool);\n            }\n','            trace.tensor(layer,"q_projection",q.data(),Q*D,{Q,D});\n            trace.tensor(layer,"k_projection",k.data(),K*D,{K,D});\n            trace.tensor(layer,"v_projection",v.data(),K*D,{K,D});\n')
  after('                    for(int j=0;j<D;++j){keys[dst+j]=Half(k[src+j]);values[dst+j]=Half(v[src+j]);}\n                }\n            }\n', '''            trace.tensor(layer,"q_rope",q.data(),Q*D,{Q,D});
            trace.tensor(layer,"k_rope",k.data(),K*D,{K,D});
            if(trace.enabled) {
                std::vector<float> canonical(size_t(Target62Trace::target+1)*K*D);
                for(int t=0;t<=Target62Trace::target;++t)for(int head=0;head<K;++head)for(int j=0;j<D;++j)
                    canonical[(size_t(t)*K+head)*D+j]=static_cast<float>(keys[cache_offset(layer,t,head)+j]);
                trace.write(layer,"canonical_keys",canonical.data(),{Target62Trace::target+1,K,D});
                for(int t=0;t<=Target62Trace::target;++t)for(int head=0;head<K;++head)for(int j=0;j<D;++j)
                    canonical[(size_t(t)*K+head)*D+j]=static_cast<float>(values[cache_offset(layer,t,head)+j]);
                trace.write(layer,"canonical_values",canonical.data(),{Target62Trace::target+1,K,D});
            }
''')
  marker='            project(tensor(p+"self_attn.o_proj.weight"),attended.data(),residual.data(),batch,pool);\n'
  assert chunk.count(marker)==1;chunk=chunk.replace(marker,'            trace.tensor(layer,"attended",attended.data(),Q*D,{Q*D});\n'+marker,1)
  after(marker+'            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];\n','            trace.tensor(layer,"post_attention_residual",x.data(),H,{H});\n')
  after('            nw=tensor(p+"post_attention_layernorm.weight").norm;for(int b=0;b<batch;++b)norm(x.data()+b*H,nw,h.data()+b*H,H);\n','            trace.tensor(layer,"post_norm_h",h.data(),H,{H});\n')
  after('            for(size_t j=0;j<gate.size();++j)gate[j]=gate[j]*(0.5f+0.5f*tanh_gate[j])*up[j];\n','            trace.tensor(layer,"activated_mlp",gate.data(),F,{F});\n')
  after('            project(tensor(p+"mlp.down_proj.weight"),gate.data(),residual.data(),batch,pool);\n            for(size_t j=0;j<x.size();++j)x[j]+=residual[j];\n','            trace.tensor(layer,"post_mlp_residual",x.data(),H,{H});\n')
  after('        norm(x.data()+size_t(batch-1)*H,tensor("model.norm.weight").norm,last_hidden.data(),H);\n','        trace.finish();\n')
  text=text[:start]+chunk+text[end:]
 (dest/name).write_text(text)
phases={'input_norm_h':[1024],'q_projection':[16,128],'k_projection':[8,128],'v_projection':[8,128],'q_rope':[16,128],'k_rope':[8,128],'canonical_keys':[63,8,128],'canonical_values':[63,8,128],'attended':[2048],'post_attention_residual':[1024],'post_norm_h':[1024],'activated_mlp':[3072],'post_mlp_residual':[1024]}
meta=dict(prepared_only=True,built=False,diagnostic_only=True,promotion_allowed=False,timing_claim=False,baseline='build-19',target_global_position=62,environment='QWEN_NATIVE_TRACE_DIR',capture_directory='capture-000000 etc, one directory per target-containing ordinary chunk, supporting independent resets/repeats',trace_manifest='manifest.jsonl; start + 28*13tensorrecords + complete; no claimedSHA until root driver verifies files',phases=phases,arithmetic='No removed/changed original arithmetic lines; observers inserted after synchronous phases. Canonical halfKV widened/reordered only in diagnostic temporary copy.',source_sha256={str(source/n):sha(source/n) for n in ('core.cpp','core.hpp','main.cpp')},candidate_sha256={str(dest/n):sha(dest/n) for n in ('core.cpp','core.hpp','main.cpp')},required_qualification='Root must prove final full-vocabulary bytes match untouchedbuild19 separately forbatched63 andsequentialB1 replay; traced output cannot be interpreted otherwise. No inference/build executed by preparer.')
(dest/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n');print(dest)
