// Original diagnostic client using public llama/ggml APIs. No inference kernels.
// Sparse evaluation callbacks alter graph scheduling: never a scoreboard driver.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}

class PersistentPools {
    using Owner=std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)>;
    Owner decode_{nullptr,ggml_threadpool_free},prefill_{nullptr,ggml_threadpool_free};
    bool separate_;
public:
    PersistentPools(int decode_threads,int prefill_threads):separate_(decode_threads!=prefill_threads) {
        auto decode_params=ggml_threadpool_params_default(decode_threads);
        auto prefill_params=ggml_threadpool_params_default(prefill_threads);
        // Public API defaults match server common CPU settings: poll50, normal
        // priority, inherited/default affinity, and no strict CPU placement.
        decode_params.poll=prefill_params.poll=50;
        decode_params.prio=prefill_params.prio=GGML_SCHED_PRIO_NORMAL;
        decode_params.strict_cpu=prefill_params.strict_cpu=false;
        prefill_params.paused=false;
        if(separate_) {
            prefill_.reset(ggml_threadpool_new(&prefill_params));
            if(!prefill_) throw std::runtime_error("Persistent prefill pool creation failed");
        }
        // In the two-pool case the first prompt uses the prefill pool. The CPU
        // backend pauses the previous pool and resumes the selected one later.
        decode_params.paused=separate_;
        decode_.reset(ggml_threadpool_new(&decode_params));
        if(!decode_) throw std::runtime_error("Persistent decode pool creation failed");
    }
    ggml_threadpool_t decode() const {return decode_.get();}
    ggml_threadpool_t prefill() const {return prefill_.get();}
    Json settings() const {
        return Json{{"attached",true},{"persistent",true},{"pool_constructions",separate_?2:1},
            {"decode_threads",ggml_threadpool_get_n_threads(decode_.get())},
            {"prefill_threads",ggml_threadpool_get_n_threads(separate_?prefill_.get():decode_.get())},
            {"shared_when_thread_counts_equal",!separate_},{"poll",50},{"priority","normal"},
            {"strict_cpu",false},{"cpu_affinity","inherited/default, not pinned"},
            {"initial_decode_paused",separate_},{"initial_prefill_paused",false},
            {"lifetime","Created once before context inference; retained across all warmups and requests"}};
    }
};
struct PoolAttachment {
    llama_context* context;
    PoolAttachment(llama_context* ctx,const PersistentPools& pools):context(ctx) {
        llama_attach_threadpool(context,pools.decode(),pools.prefill());
    }
    ~PoolAttachment() {
        llama_synchronize(context);
        llama_detach_threadpool(context);
    }
};

struct HeadSpan { int step; double begin, end; };
struct CallbackState {
    Clock::time_point origin{}, begin{};
    int step=0;
    bool pending=false, invalid=false;
    size_t count=0;
    std::vector<HeadSpan> storage;
};
static bool boundary_callback(ggml_tensor* tensor, bool ask, void* opaque) {
    auto& state=*static_cast<CallbackState*>(opaque);
    const bool norm=std::strcmp(tensor->name,"result_norm")==0;
    const bool output=std::strcmp(tensor->name,"result_output")==0;
    if(ask) return norm || output;
    const auto now=Clock::now();
    if(norm) {
        if(state.pending) state.invalid=true;
        state.begin=now;state.pending=true;
    } else if(output) {
        if(!state.pending || state.count==state.storage.size()) state.invalid=true;
        else state.storage[state.count++]={state.step,seconds(state.origin,state.begin),seconds(state.origin,now)};
        state.pending=false;
    } else state.invalid=true;
    return true;
}

static std::vector<llama_token> read_ids(const std::string& path) {
    std::ifstream file(path);
    if(!file) throw std::runtime_error("Cannot open token-ID JSON: "+path);
    Json data;file>>data;
    if(!data.is_array()) throw std::runtime_error("Token-ID file must be a JSON integer array");
    std::vector<llama_token> result;
    for(const auto& item:data) {
        if(!item.is_number_integer() || item.get<int64_t>()<0 || item.get<int64_t>()>=151936)
            throw std::runtime_error("Invalid Qwen3 token ID");
        result.push_back(item.get<llama_token>());
    }
    return result;
}
static std::string metadata(const llama_model* model,const char* key) {
    char text[512]{};
    const int length=llama_model_meta_val_str(model,key,text,sizeof(text));
    if(length<0) return "";
    if(length>=int(sizeof(text))) throw std::runtime_error("Model metadata too long");
    return text;
}
static void save_json(const fs::path& path,const Json& report) {
    std::ofstream file(path);
    if(!file) throw std::runtime_error("Cannot create report");
    file<<report.dump(2)<<'\n';
    file.close();
    if(!file) throw std::runtime_error("Cannot write report");
}
static Json dump_logits(llama_context* context,const fs::path& directory,int run,int step,size_t context_tokens) {
    const float* logits=llama_get_logits_ith(context,-1);
    if(!logits) throw std::runtime_error("Missing logits");
    for(int i=0;i<151936;i++) if(!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite logits");
    const fs::path path=directory/("run-"+std::to_string(run)+"-step-"+std::to_string(step)+".f32");
    if(fs::exists(path)) throw std::runtime_error("Refusing to overwrite logit dump");
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(logits),151936*sizeof(float));
    file.close();
    if(!file) throw std::runtime_error("Logit dump failed");
    return Json{{"step",step},{"path",path.string()},{"dtype","float32"},{"byte_order","little"},
                {"shape",{151936}},{"bytes",151936*sizeof(float)},{"context_tokens",context_tokens}};
}

int main(int argc,char** argv) {
    Json report{{"completed",false},{"diagnostic_only",true},{"client_version",2},{"gpu",false},{"pytorch",false}};
    fs::path output;
    bool output_owned=false;
    try {
        std::map<std::string,std::string> args;
        const std::vector<std::string> allowed={"--model","--prompt-ids","--continuation-ids","--threads",
            "--batch-threads","--tokens","--warmups","--repetitions","--callback","--flash-attn",
            "--context","--batch-size","--output","--dump-logits-dir"};
        for(int i=1;i<argc;i+=2) {
            if(i+1>=argc || std::find(allowed.begin(),allowed.end(),argv[i])==allowed.end() || args.count(argv[i]))
                throw std::runtime_error("Unknown, repeated, or missing command-line argument");
            args.emplace(argv[i],argv[i+1]);
        }
        auto get=[&](const std::string& key,const std::string& fallback="") {
            const auto it=args.find(key);return it==args.end()?fallback:it->second;
        };
        auto integer=[&](const std::string& key,int fallback,int low,int high) {
            const std::string text=get(key,std::to_string(fallback));size_t used=0;
            const int value=std::stoi(text,&used);
            if(used!=text.size() || value<low || value>high) throw std::runtime_error("Invalid "+key);
            return value;
        };
        output=get("--output");
        if(output.empty() || get("--model").empty() || get("--prompt-ids").empty())
            throw std::runtime_error("--model, --prompt-ids and --output are required");
        if(fs::exists(output)) throw std::runtime_error("Refusing to overwrite report");
        if(!output.parent_path().empty()) fs::create_directories(output.parent_path());
        output_owned=true;
        const int threads=integer("--threads",8,1,64),batch_threads=integer("--batch-threads",12,1,64);
        const int tokens=integer("--tokens",32,2,512),warmups=integer("--warmups",1,0,10);
        const int repetitions=integer("--repetitions",3,1,100),context=integer("--context",4096,32,4096);
        const int batch_size=integer("--batch-size",512,1,4096);
        const std::string callback=get("--callback","off"),flash=get("--flash-attn","auto");
        if(callback!="off" && callback!="on") throw std::runtime_error("--callback must be off or on");
        if(flash!="off" && flash!="on" && flash!="auto") throw std::runtime_error("Invalid --flash-attn");
        auto prompt=read_ids(get("--prompt-ids"));
        const bool teacher_forced=args.count("--continuation-ids");
        auto continuation=teacher_forced?read_ids(get("--continuation-ids")):std::vector<llama_token>{};
        if(prompt.empty() || prompt.size()>size_t(batch_size) || prompt.size()+tokens-1>size_t(context))
            throw std::runtime_error("Prompt must fit one physical batch and the requested context");
        if(teacher_forced && continuation.size()!=size_t(tokens-1))
            throw std::runtime_error("Continuation must contain exactly --tokens minus one consumed IDs");
        const fs::path dump_dir=get("--dump-logits-dir");
        if(!dump_dir.empty()) fs::create_directories(dump_dir);
        const uint16_t endian=1;
        if(*reinterpret_cast<const uint8_t*>(&endian)!=1 || sizeof(float)!=4)
            throw std::runtime_error("Dump format requires little-endian binary32 host");

        llama_backend_init();
        struct BackendLifetime {~BackendLifetime(){llama_backend_free();}} backend_lifetime;
        Json devices=Json::array();
        for(size_t i=0;i<ggml_backend_dev_count();i++) {
            const auto device=ggml_backend_dev_get(i);
            const auto type=ggml_backend_dev_type(device);
            if(type==GGML_BACKEND_DEVICE_TYPE_GPU || type==GGML_BACKEND_DEVICE_TYPE_IGPU || type==GGML_BACKEND_DEVICE_TYPE_META)
                throw std::runtime_error("Non-CPU device registered; use a CPU-only build");
            devices.push_back(Json{{"name",ggml_backend_dev_name(device)},{"type",int(type)}});
        }
        llama_model_params model_params=llama_model_default_params();
        ggml_backend_dev_t no_offload_devices[]={nullptr};
        model_params.devices=no_offload_devices;model_params.n_gpu_layers=0;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(get("--model").c_str(),model_params),llama_model_free);
        if(!model) throw std::runtime_error("Model load failed");
        const auto vocab=llama_model_get_vocab(model.get());
        if(metadata(model.get(),"general.architecture")!="qwen3" || llama_model_n_layer(model.get())!=28 ||
           llama_model_n_embd(model.get())!=1024 || llama_model_n_head(model.get())!=16 ||
           llama_model_n_head_kv(model.get())!=8 || llama_vocab_n_tokens(vocab)!=151936)
            throw std::runtime_error("This diagnostic requires Qwen3-0.6B dimensions");
        CallbackState state;state.storage.resize(tokens+2);
        auto params=llama_context_default_params();
        params.n_ctx=context;params.n_batch=batch_size;params.n_ubatch=batch_size;params.n_seq_max=1;
        params.n_threads=threads;params.n_threads_batch=batch_threads;
        params.type_k=GGML_TYPE_F16;params.type_v=GGML_TYPE_F16;
        params.offload_kqv=false;params.op_offload=false;params.no_perf=false;
        params.flash_attn_type=flash=="on"?LLAMA_FLASH_ATTN_TYPE_ENABLED:
            flash=="off"?LLAMA_FLASH_ATTN_TYPE_DISABLED:LLAMA_FLASH_ATTN_TYPE_AUTO;
        params.cb_eval=callback=="on"?boundary_callback:nullptr;
        params.cb_eval_user_data=&state;
        // Declaration order guarantees: detach, destroy context, then free pools.
        // The context's CPU backend can retain its last-selected pool pointer.
        PersistentPools pools(threads,batch_threads);
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),params),llama_free);
        if(!ctx) throw std::runtime_error("Context creation failed");
        PoolAttachment attachment(ctx.get(),pools);
        auto sampler_params=llama_sampler_chain_default_params();sampler_params.no_perf=false;
        std::unique_ptr<llama_sampler,decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(sampler_params),llama_sampler_free);
        if(!sampler) throw std::runtime_error("Sampler creation failed");
        auto* greedy=llama_sampler_init_greedy();
        if(!greedy) throw std::runtime_error("Greedy sampler creation failed");
        llama_sampler_chain_add(sampler.get(),greedy);
        report["configuration"]={{"model",get("--model")},{"model_file_type",metadata(model.get(),"general.file_type")},
            {"architecture","qwen3"},{"layers",28},{"hidden",1024},{"query_heads",16},{"kv_heads",8},{"vocabulary",151936},
            {"threads",llama_n_threads(ctx.get())},{"batch_threads",llama_n_threads_batch(ctx.get())},
            {"context",llama_n_ctx(ctx.get())},{"batch_size",batch_size},{"kv_type","f16"},{"flash_attn_requested",flash},
            {"callback",callback},{"registered_devices",devices},{"n_gpu_layers",0},{"offload_kqv",false},{"op_offload",false}};
        report["configuration"]["threadpools"]=pools.settings();
        report["prompt_ids"]=prompt;report["continuation_ids"]=continuation;
        report["mode"]=teacher_forced?"teacher_forced":"greedy";
        report["requested_steps"]=tokens;report["warmups"]=warmups;report["runs"]=Json::array();
        report["limitations"]={
            "Profiling diagnostic only; sparse callbacks split graphs and synchronize the backend, adding overhead.",
            "Head windows are result_norm completion to result_output completion, including intervening dispatch/synchronization; not pure kernel time.",
            "Decode step sums include decode, synchronization and greedy sampling; exclude inter-step bookkeeping and optional full-logit disk writes.",
            "Dumping logits occurs outside timed steps but changes cache/timing conditions. Dump-enabled results are numerical diagnostics only.",
            "Teacher-forced mode consumes the provided IDs even if a greedy prediction is EOS; greedy mode stops naturally and rejects a shortened fixed workload.",
            "CPU worker placement is not pinned; this client does not measure per-thread CPU time or full application/server overhead."};
        std::vector<llama_token> reference;
        for(int run=0;run<warmups+repetitions;run++) {
            llama_memory_clear(llama_get_memory(ctx.get()),false);
            llama_sampler_reset(sampler.get());llama_perf_context_reset(ctx.get());
            state.pending=false;state.invalid=false;state.count=0;state.step=0;
            Json row{{"index",run},{"warmup",run<warmups},{"logit_dumps",Json::array()}};
            std::vector<llama_token> ids,consumed;ids.reserve(tokens);consumed.reserve(tokens-1);
            std::vector<int> eog_steps;
            double decode_seconds=0;
            // Correlate monotonic relative windows with external profiler wall
            // timestamps. Bracket the system-clock read; do not assume the two
            // clocks share an epoch or silently discard correlation uncertainty.
            const auto anchor_before=Clock::now();
            const auto anchor_wall=std::chrono::system_clock::now();
            const auto anchor_after=Clock::now();
            state.origin=Clock::now();
            if(llama_decode(ctx.get(),llama_batch_get_one(prompt.data(),int(prompt.size())))!=0)
                throw std::runtime_error("Prompt decode failed");
            llama_synchronize(ctx.get());
            const auto prefilled=Clock::now();
            llama_token next=llama_sampler_sample(sampler.get(),ctx.get(),-1);
            const auto first=Clock::now();
            ids.push_back(next);
            if(llama_vocab_is_eog(vocab,next)) eog_steps.push_back(0);
            if(!dump_dir.empty()) row["logit_dumps"].push_back(dump_logits(ctx.get(),dump_dir,run,0,prompt.size()));
            for(int step=1;step<tokens;step++) {
                if(!teacher_forced && llama_vocab_is_eog(vocab,next)) break;
                llama_token input=teacher_forced?continuation[step-1]:next;
                consumed.push_back(input);state.step=step;
                const auto begin=Clock::now();
                if(llama_decode(ctx.get(),llama_batch_get_one(&input,1))!=0) throw std::runtime_error("Token decode failed");
                llama_synchronize(ctx.get());
                next=llama_sampler_sample(sampler.get(),ctx.get(),-1);
                const auto end=Clock::now();decode_seconds+=seconds(begin,end);
                ids.push_back(next);
                if(llama_vocab_is_eog(vocab,next)) eog_steps.push_back(step);
                if(!dump_dir.empty()) row["logit_dumps"].push_back(dump_logits(ctx.get(),dump_dir,run,step,prompt.size()+step));
            }
            const auto completed=Clock::now();
            if(callback=="on" && (state.invalid || state.pending || state.count!=ids.size()))
                throw std::runtime_error("Unexpected sparse callback boundary sequence");
            row["predicted_token_ids"]=ids;row["consumed_continuation_ids"]=consumed;row["eog_prediction_steps"]=eog_steps;
            row["prefill_seconds"]=seconds(state.origin,prefilled);row["ttft_seconds"]=seconds(state.origin,first);
            row["decode_step_seconds"]=decode_seconds;row["decode_steps"]=consumed.size();
            row["decode_step_tokens_per_second"]=decode_seconds>0?double(consumed.size())/decode_seconds:0;
            row["inference_step_seconds"]=seconds(state.origin,first)+decode_seconds;
            row["request_wall_including_dumps_seconds"]=seconds(state.origin,completed);
            row["clock_correlation"]={{"unix_time_ns",std::chrono::duration_cast<std::chrono::nanoseconds>(anchor_wall.time_since_epoch()).count()},
                {"steady_before_seconds",seconds(state.origin,anchor_before)},
                {"steady_after_seconds",seconds(state.origin,anchor_after)},
                {"meaning","System-clock timestamp sampled between the two monotonic offsets relative to this request origin"}};
            row["phase_windows"]={{"prefill_begin_seconds",0.0},{"prefill_end_seconds",seconds(state.origin,prefilled)},
                {"decode_wall_begin_seconds",seconds(state.origin,first)},
                {"decode_wall_end_seconds",seconds(state.origin,completed)},
                {"decode_wall_includes_optional_dumps_and_bookkeeping",true}};
            row["head_windows"]=Json::array();double head_prefill=0,head_decode=0;
            for(size_t i=0;i<state.count;i++) {
                const auto& span=state.storage[i];const double elapsed=span.end-span.begin;
                row["head_windows"].push_back(Json{{"step",span.step},{"phase",span.step==0?"prefill":"decode"},
                    {"begin_seconds",span.begin},{"end_seconds",span.end},{"elapsed_seconds",elapsed}});
                (span.step==0?head_prefill:head_decode)+=elapsed;
            }
            row["head_prefill_window_seconds"]=callback=="on"?Json(head_prefill):Json(nullptr);
            row["head_decode_window_seconds"]=callback=="on"?Json(head_decode):Json(nullptr);
            const auto perf=llama_perf_context(ctx.get());
            row["llama_perf"]={{"prompt_ms",perf.t_p_eval_ms},{"decode_ms",perf.t_eval_ms},
                {"prompt_tokens",perf.n_p_eval},{"decode_tokens",perf.n_eval},{"graph_reuses",perf.n_reused},
                {"reallocations_same_shape",perf.n_reallocs_same},{"reallocations_changed_shape",perf.n_reallocs_diff}};
            row["fixed_work_valid"]=ids.size()==size_t(tokens) && (teacher_forced || eog_steps.empty());
            if(run==0) reference=ids;
            row["own_repeat_ids_equal"]=ids==reference;
            report["runs"].push_back(row);save_json(output,report);
            if(!row["fixed_work_valid"].get<bool>()) throw std::runtime_error("Natural EOS invalidated requested fixed greedy workload");
            if(ids!=reference) throw std::runtime_error("Repeated predictions changed");
        }
        report["completed"]=true;save_json(output,report);
        return 0;
    } catch(const std::exception& error) {
        report["error"]=error.what();
        if(output_owned) {try {save_json(output,report);}catch(...) {}}
        std::cerr<<"profile_native_cpu: "<<error.what()<<'\n';
        return 1;
    }
}
