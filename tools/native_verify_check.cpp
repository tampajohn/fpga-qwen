// API-only exact verification gate. No CLI sampler or speculative decoding.
#include "core.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using qwen_cpu::Model;
constexpr int H=1024,V=qwen_cpu::vocabulary;
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void same(std::span<const float> a,std::span<const float> b,const std::string& why){
    require(a.size()==b.size(),why+" shape");
    for(float x:a)require(std::isfinite(x),why+" nonfinite");
    for(float x:b)require(std::isfinite(x),why+" nonfinite reference");
    require(a.empty()||std::memcmp(a.data(),b.data(),a.size_bytes())==0,why+" bits");
}
struct State {int position;std::vector<float> logits,hidden;};
State state(Model& m){
    State s{m.position(),m.logits(),{}};
    if(s.logits.empty()){
        bool rejected=false;try{(void)m.hidden();}catch(const std::exception&){rejected=true;}
        require(rejected,"Hidden accessible without current logits");
    }else{
        auto h=m.hidden();s.hidden.assign(h.begin(),h.end());
        require(s.logits.size()==V&&s.hidden.size()==H,"Current array shape");
    }
    return s;
}
void unchanged(Model& m,const State& expected){auto actual=state(m);
    require(actual.position==expected.position,"Public position changed");
    same(actual.logits,expected.logits,"Public logits");same(actual.hidden,expected.hidden,"Public hidden");
}
void equal_models(Model& a,Model& b){unchanged(a,state(b));}
std::vector<int> prefix(int n){std::vector<int> ids(n);for(int i=0;i<n;++i)ids[i]=32+(i*37)%1000;return ids;}
void prepare(Model& m,int n){m.reset();if(n)m.prefill(prefix(n));}
template<class Fn>void rejects(Model& m,Fn fn,const State& expected){
    bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}
    require(rejected,"Expected validation rejection");unchanged(m,expected);
}
void row(std::span<const float> logits,std::span<const float> hidden,int i,const State& reference){
    same(logits.subspan(size_t(i)*V,V),reference.logits,"Verification logits row"+std::to_string(i));
    same(hidden.subspan(size_t(i)*H,H),reference.hidden,"Verification hidden row"+std::to_string(i));
}
void normal_case(Model& candidate,Model& oracle,int n,int commit){
    prepare(candidate,n);prepare(oracle,n);equal_models(candidate,oracle);
    const auto base=state(candidate);
    // Validate before mutation, including a valid first and invalid second ID.
    rejects(candidate,[&]{candidate.verify_two(-1,11);},base);
    rejects(candidate,[&]{candidate.verify_two(9707,V);},base);
    rejects(candidate,[&]{candidate.commit_verified(1);},base);
    auto pending=candidate.verify_two(9707,11);
    require(pending.base_position==n&&pending.logits.size()==2*V&&pending.hidden.size()==2*H,"Pending view shape/base");
    unchanged(candidate,base);
    oracle.step(9707);const auto first=state(oracle);row(pending.logits,pending.hidden,0,first);
    oracle.step(11);const auto second=state(oracle);row(pending.logits,pending.hidden,1,second);
    const std::vector<float> saved_logits(pending.logits.begin(),pending.logits.end());
    const std::vector<float> saved_hidden(pending.hidden.begin(),pending.hidden.end());
    auto intact=[&]{unchanged(candidate,base);same(pending.logits,saved_logits,"Pending logits preserved");same(pending.hidden,saved_hidden,"Pending hidden preserved");};
    rejects(candidate,[&]{candidate.step(12);},base);intact();
    rejects(candidate,[&]{candidate.prefill({12});},base);intact();
    rejects(candidate,[&]{candidate.verify_two(12,13);},base);intact();
    for(int count:{-1,0,3}){rejects(candidate,[&]{candidate.commit_verified(count);},base);intact();}
    for(int pos:{-1,n+1}){rejects(candidate,[&]{candidate.truncate(pos);},base);intact();}
    candidate.commit_verified(commit);
    unchanged(candidate,commit==1?first:second);
    // Never dereference borrowed pending views after commit.
    if(commit==1)oracle.truncate(n+1);
    candidate.step(42);oracle.step(42);equal_models(candidate,oracle);
    candidate.step(99);oracle.step(99);equal_models(candidate,oracle);
}
void edge_cases(const std::string& weights,int threads){
    qwen_cpu::Options opts{threads,6,4};Model c(weights,opts),o(weights,opts);
    prepare(c,3);const auto before=state(c);
    rejects(c,[&]{c.verify_two(9707,11);},before); // Only one slot remains.
    c.step(42);const auto full=state(c);
    rejects(c,[&]{c.verify_two(9707,11);},full);rejects(c,[&]{c.step(11);},full);
    prepare(c,2);prepare(o,2);c.verify_two(9707,11);c.commit_verified(2);
    o.step(9707);o.step(11);equal_models(c,o);
    const auto committed=state(c);rejects(c,[&]{c.commit_verified(1);},committed);
    // Abort a pending transaction, retaining exactly the requested base prefix.
    for(int retained:{0,1,2}){
        prepare(c,2);prepare(o,2);c.verify_two(9707,11);c.truncate(retained);o.truncate(retained);
        equal_models(c,o);require(c.logits().empty(),"Truncate must invalidate logits");
        rejects(c,[&]{c.commit_verified(1);},state(c));
        c.step(42);o.step(42);equal_models(c,o);
    }
    prepare(c,2);c.verify_two(9707,11);c.reset();o.reset();equal_models(c,o);
    rejects(c,[&]{c.commit_verified(2);},state(c));
    auto pending=c.verify_two(9707,11);o.step(9707);row(pending.logits,pending.hidden,0,state(o));
    o.step(11);row(pending.logits,pending.hidden,1,state(o));c.commit_verified(2);equal_models(c,o);
}
int main(int argc,char** argv){
    try{
        require(argc==3||argc==4,"Usage: native_verify_check WEIGHTS WORKERS [--smoke]");
        const bool smoke=argc==4;require(!smoke||std::string(argv[3])=="--smoke","Unknown argument");
        const std::string weights=argv[1];const int workers=std::stoi(argv[2]);
        require(workers==1||workers==12||workers==64,"Predeclared workers are1,12,64");
        {
            qwen_cpu::Options options{workers,6,4096};Model candidate(weights,options),oracle(weights,options);
            const std::vector<int> prefixes=smoke?std::vector<int>{0,1}:std::vector<int>{0,1,511,512,513,3966};
            for(int n:prefixes)for(int commit:{1,2}){
                std::cout<<"case prefix="<<n<<" commit="<<commit<<std::endl;
                normal_case(candidate,oracle,n,commit);
            }
        }
        edge_cases(weights,workers);
        std::cout<<"{\"completed\":true,\"passed\":true,\"workers\":"<<workers
                 <<",\"prefill_workers\":6,\"prefix_commit_cases\":"<<(smoke?4:12)<<",\"edge_cases_passed\":true} "<<std::endl;
        return 0;
    }catch(const std::exception& error){std::cerr<<"Verification gate failed: "<<error.what()<<std::endl;return 1;}
}
