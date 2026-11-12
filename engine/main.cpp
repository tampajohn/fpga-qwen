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
    int select(const std::vector<float>& logits) {
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
           "\"activation_precision\":\"f32\",\"kv_precision\":\"f16\",\"prefix_reuse\":false,\"suffix_drafting\":false}\n" << std::flush;
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
        settings(options,model,true);
        std::string line;
        while(std::getline(std::cin,line)) {
            try {
                if(line.size()>1024*1024 || line.find('\0')!=std::string::npos)
                    throw std::runtime_error("Invalid command line");
                std::istringstream input(line);std::string command;input>>command;
                if(command=="quit") { no_extra(input);std::cout<<"{\"ok\":true}\n"<<std::flush;break; }
                if(command=="reset") {
                    no_extra(input);model.reset();sampler.reset();
                    std::cout<<"{\"ok\":true,\"position\":0}\n";
                } else if(command=="mode") {
                    std::string mode,seed;input>>mode;
                    if(mode=="greedy") {no_extra(input);sampler.mode(false,0);}
                    else if(mode=="chat") {if(!(input>>seed))throw std::runtime_error("Chat seed required");no_extra(input);sampler.mode(true,integer<uint64_t>(seed));}
                    else throw std::runtime_error("Unknown decoding mode");
                    std::cout<<"{\"ok\":true}\n";
                } else if(command=="prefill" || command=="step") {
                    std::vector<int> ids;std::string token;
                    while(input>>token) ids.push_back(integer<int>(token));
                    if(ids.empty() || (command=="step" && ids.size()!=1)) throw std::runtime_error("Invalid token count");
                    for(int id:ids) if(id<0 || id>=qwen_cpu::vocabulary) throw std::runtime_error("Token outside vocabulary");
                    if(ids.size()>size_t(options.context-model.position()))throw std::runtime_error("Context capacity exceeded");
                    const auto start=Clock::now();
                    if(command=="step") model.step(ids[0]); else model.prefill(ids);
                    for(int id:ids) sampler.consumed(id);
                    const int selected=sampler.select(model.logits());
                    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
                    std::cout<<"{\"token\":"<<selected<<",\"position\":"<<model.position()<<",\"seconds\":"<<std::setprecision(17)<<elapsed<<"}\n";
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
                model.reset();sampler.reset();
                std::cout<<"{\"error\":"<<quote(error.what())<<",\"position\":0}\n";
            }
            std::cout<<std::flush;
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"qwen_cpu: "<<error.what()<<'\n';return 1;
    }
}
