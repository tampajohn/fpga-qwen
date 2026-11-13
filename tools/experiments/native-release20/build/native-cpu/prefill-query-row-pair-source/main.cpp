#include "core.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
template<class T> T integer(const std::string& text) {
    T result{};
    const auto parsed = std::from_chars(text.data(), text.data()+text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data()+text.size() || text.empty())
        throw std::runtime_error("Invalid integer: " + text);
    return result;
}
void no_extra(std::istringstream& input) {
    std::string extra;
    if (input >> extra) throw std::runtime_error("Unexpected command argument");
}
class Sampler {
    struct Score { float value; int id; };
    std::vector<unsigned char> seen_{std::vector<unsigned char>(qwen_cpu::vocabulary, 0)};
    std::vector<Score> scores_{std::vector<Score>(qwen_cpu::vocabulary)};
    std::mt19937_64 random_{0};
    bool chat_ = false;
public:
    void reset() { std::fill(seen_.begin(),seen_.end(),0); }
    void mode(bool chat, uint64_t seed) { chat_=chat; random_.seed(seed); }
    void consumed(int id) { seen_.at(id)=1; }
    bool greedy_mode()const{return !chat_;}
    int select(std::span<const float> logits) {
        if (logits.size()!=qwen_cpu::vocabulary) throw std::runtime_error("No current full-vocabulary logits");
        int best=0;
        for(int id=0;id<qwen_cpu::vocabulary;id++) {
            if(!std::isfinite(logits[id])) throw std::runtime_error("Nonfinite model logits");
            if(logits[id]>logits[best]) best=id;
            if(chat_) scores_[id]={(logits[id]-(seen_[id]?1.5f:0.f))/.7f,id};
        }
        if(!chat_) return best;
        std::partial_sort(scores_.begin(),scores_.begin()+20,scores_.end(),
            [](const Score& a,const Score& b){return a.value>b.value || (a.value==b.value && a.id<b.id);});
        double probabilities[20],total=0;
        for(int i=0;i<20;i++) total+=(probabilities[i]=std::exp(double(scores_[i].value)-scores_[0].value));
        double retained=0; int count=0;
        do { retained+=probabilities[count++]; } while(count<20 && retained<.8*total);
        double draw=std::generate_canonical<double,53>(random_)*retained;
        for(int i=0;i<count;i++) if((draw-=probabilities[i])<0) return scores_[i].id;
        return scores_[count-1].id;
    }
};
void settings(const qwen_cpu::Options& options, const qwen_cpu::Model& model, bool ready) {
    const int position=model.position();
    std::cout << "{\"ready\":" << (ready?"true":"false") << ",\"protocol\":1,\"position\":" << position
        << ",\"decode_threads\":" << options.decode_threads << ",\"prefill_threads\":" << options.prefill_threads
        << ",\"context\":" << options.context
        << ",\"lossless_bf16_decode_cache_bytes\":" << model.lossless_bf16_cache_bytes()
        << ",\"gpu\":false,\"pytorch\":false,\"weight_precision\":\"f16\",\"norm_precision\":\"f32\","
           "\"norm_accumulator_precision\":\"f64\",\"activation_precision\":\"f32\",\"kv_precision\":\"f16\",\"prefix_reuse\":true,\"prefix_reuse_chunk\":" << qwen_cpu::prefix_reuse_chunk << ",\"suffix_drafting\":false,\"verify_greedy_b2\":true,\"max_verified_draft\":1}\n" << std::flush;
}
}
int main(int argc,char** argv) {
    try {
        qwen_cpu::Options options;
        std::string weights;
        for(int i=1;i<argc;i++) {
            std::string key=argv[i];
            if(i+1==argc) throw std::runtime_error("Missing option value");
            std::string value=argv[++i];
            if(key=="--weights") weights=value;
            else if(key=="--threads") options.decode_threads=integer<int>(value);
            else if(key=="--prefill-threads") options.prefill_threads=integer<int>(value);
            else if(key=="--context") options.context=integer<int>(value);
            else throw std::runtime_error("Unknown option: "+key);
        }
        if(weights.empty()) throw std::runtime_error("--weights is required");
        qwen_cpu::Model model(weights,options);
        Sampler sampler;
        std::vector<int> consumed_history;consumed_history.reserve(options.context);
        settings(options,model,true);
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                if(line.size()>1024*1024 || line.find('\0')!=std::string::npos)
                    throw std::runtime_error("Invalid command line");
                std::istringstream input(line);std::string command;input>>command;
                if(command=="quit") { no_extra(input);std::cout<<"{\"ok\":true}\n"<<std::flush;break; }
                if(command=="reset") {
                    no_extra(input);model.reset();sampler.reset();consumed_history.clear();
                    std::cout<<"{\"ok\":true,\"position\":0}\n";
                } else if(command=="rewind") {
                    std::string target;
                    if(!(input>>target))throw std::runtime_error("Rewind position required");
                    no_extra(input);
                    const int retained=integer<int>(target);
                    if(consumed_history.size()!=size_t(model.position()))throw std::runtime_error("Consumed history does not match model position");
                    // Model validates the target before any KV/logit mutation.
                    model.truncate(retained);
                    consumed_history.resize(size_t(retained));
                    sampler.reset();
                    for(int id:consumed_history)sampler.consumed(id);
                    std::cout<<"{\"ok\":true,\"position\":"<<retained<<"}\n";
                } else if(command=="mode") {
                    std::string mode,seed;input>>mode;
                    if(mode=="greedy") {no_extra(input);sampler.mode(false,0);}
                    else if(mode=="chat") {if(!(input>>seed))throw std::runtime_error("Chat seed required");no_extra(input);sampler.mode(true,integer<uint64_t>(seed));}
                    else throw std::runtime_error("Unknown decoding mode");
                    std::cout<<"{\"ok\":true}\n";
                } else if(command=="verify-greedy") {
                    std::string pending_text,draft_text;
                    if(!(input>>pending_text>>draft_text))throw std::runtime_error("verify-greedy requires pending and draft IDs");
                    no_extra(input);
                    const int pending=integer<int>(pending_text),draft=integer<int>(draft_text);
                    if(!sampler.greedy_mode())throw std::runtime_error("verify-greedy is unavailable in chat mode");
                    if(consumed_history.size()!=size_t(model.position()))throw std::runtime_error("Consumed history does not match model position");
                    const auto start=Clock::now();
                    const auto rows=model.verify_two(pending,draft);
                    const auto eos=[](int id){return id==151643||id==151645;};
                    const int first=sampler.select(rows.logits.first(qwen_cpu::vocabulary));
                    const bool accepted=first==draft;
                    const int committed=accepted&&!eos(first)?2:1;
                    int second=-1;
                    if(committed==2)second=sampler.select(rows.logits.subspan(qwen_cpu::vocabulary,qwen_cpu::vocabulary));
                    // Correction/bonus predictions remain unconsumed. Commit
                    // only inputs actually traversed by sequential greedy decode.
                    model.commit_verified(committed);
                    consumed_history.push_back(pending);sampler.consumed(pending);
                    if(committed==2){consumed_history.push_back(draft);sampler.consumed(draft);}
                    const bool stopped=eos(first)||(committed==2&&eos(second));
                    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
                    std::cout<<"{\"tokens\":["<<first;
                    if(committed==2)std::cout<<","<<second;
                    std::cout<<"],\"accepted\":"<<(accepted?1:0)<<",\"committed_inputs\":"<<committed
                        <<",\"position\":"<<model.position()<<",\"verified_positions\":2,\"finish_reason\":"
                        <<(stopped?"\"eos\"":"null")<<",\"seconds\":"<<std::setprecision(17)<<elapsed<<"}\n";
                } else if(command=="prefill" || command=="step") {
                    std::vector<int> ids;std::string token;
                    while(input>>token) ids.push_back(integer<int>(token));
                    if(ids.empty() || (command=="step" && ids.size()!=1)) throw std::runtime_error("Invalid token count");
                    for(int id:ids) if(id<0 || id>=qwen_cpu::vocabulary) throw std::runtime_error("Token outside vocabulary");
                    if(ids.size()>size_t(options.context-model.position()))throw std::runtime_error("Context capacity exceeded");
                    if(consumed_history.size()!=size_t(model.position()))throw std::runtime_error("Consumed history does not match model position");
                    const auto start=Clock::now();
                    if(command=="step") model.step(ids[0]); else model.prefill(ids);
                    consumed_history.insert(consumed_history.end(),ids.begin(),ids.end());
                    for(int id:ids) sampler.consumed(id);
                    const int selected=sampler.select(model.logits());
                    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
                    std::cout<<"{\"token\":"<<selected<<",\"position\":"<<model.position()<<",\"seconds\":"<<std::setprecision(17)<<elapsed<<"}\n";
                } else if(command=="dump-hidden") {
                    const auto begin=line.find(' ');
                    if(begin==std::string::npos || begin+1==line.size())throw std::runtime_error("Hidden dump path required");
                    const auto hidden=model.hidden();
                    std::ofstream out(line.substr(begin+1),std::ios::binary|std::ios::trunc);
                    if(!out)throw std::runtime_error("Cannot open hidden dump path");
                    out.write(reinterpret_cast<const char*>(hidden.data()),hidden.size_bytes());out.close();
                    if(!out)throw std::runtime_error("Failed writing hidden dump");
                    std::cout<<"{\"ok\":true,\"count\":"<<hidden.size()<<",\"position\":"<<model.position()<<",\"format\":\"f32le\"}\n";
                } else if(command=="dump") {
                    const auto begin=line.find(' ');
                    if(begin==std::string::npos || begin+1==line.size())throw std::runtime_error("Dump path required");
                    const auto& logits=model.logits();
                    if(logits.size()!=qwen_cpu::vocabulary || model.position()==0)throw std::runtime_error("No logits to dump");
                    std::ofstream out(line.substr(begin+1),std::ios::binary|std::ios::trunc);
                    if(!out)throw std::runtime_error("Cannot open dump path");
                    out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));out.close();
                    if(!out)throw std::runtime_error("Failed writing logits dump");
                    std::cout<<"{\"ok\":true,\"count\":"<<qwen_cpu::vocabulary<<",\"position\":"<<model.position()<<"}\n";
                } else if(command=="stats") {no_extra(input);settings(options,model,true);continue;}
                else throw std::runtime_error("Unknown command");
            } catch(const std::exception& error) {
                model.reset();sampler.reset();consumed_history.clear();
                std::cout<<"{\"error\":"<<quote(error.what())<<",\"position\":0}\n";
            }
            std::cout<<std::flush;
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"qwen_cpu: "<<error.what()<<'\n';return 1;
    }
}
