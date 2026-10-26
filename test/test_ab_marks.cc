#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <random>
#include <chrono>
using namespace ngfixlib;
static void check(bool x,const char* what) { if(!x) throw std::runtime_error(what); }
static void initialize(HNSW_NGFix<float>& g,size_t n) {
    std::memset(g.vecdata,0,g.max_elements*g.size_per_element); g.n=n;
    for(size_t u=0;u<n;++u) { float x[16]={}; x[0]=float(u)*.1f; g.SetData(u,x); }
}
int main() {
    // c=0, m=5, n=1. Alternate in-region path 0->2->3->1 is optional.
    for(bool alternative:{false,true}) {
        HNSW_NGFix<float> g(L2_float,16,6,16,48); initialize(g,6);
        g.Graph[0].add_base_graph_neighbors(5);
        g.Graph[5].add_base_graph_neighbors(0); // reverse edge: never a fresh exit
        g.Graph[5].add_base_graph_neighbors(1);
        if(alternative) {
            g.Graph[0].add_base_graph_neighbors(2); g.Graph[2].add_base_graph_neighbors(3);
            g.Graph[3].add_base_graph_neighbors(1);
        }
        auto marks=ABMarks::layout(g); const auto e=marks.offsets.back();
        std::vector<uint32_t> op(e),a(e),b(e),excluded(6); std::vector<int32_t> rank(6,-1);
        uint32_t stamp=0; const std::vector<uint32_t> top={0,1,2,3};
        labelABQuery(g,top,marks,rank,excluded,stamp,op,a,b);
        check(op[0]==1 && a[0]==1 && b[0]==unsigned(!alternative),"Alternative path A/B labels");
        labelABQuery(g,top,marks,rank,excluded,stamp,op,a,b);
        check(op[0]==2 && a[0]==2 && b[0]==2*unsigned(!alternative),"Query accumulation/reset");
        // Now n is a direct neighbor: neither n nor the reverse edge may count.
        g.Graph[0].add_base_graph_neighbors(1); marks=ABMarks::layout(g);
        op.assign(marks.offsets.back(),0); a=op; b=op;
        labelABQuery(g,top,marks,rank,excluded,stamp,op,a,b);
        check(a[0]==0 && b[0]==0,"Exclude cur and all direct neighbors");
    }
    std::mt19937 rng(17);
    HNSW_NGFix<float> g(L2_float,16,96,16,48); initialize(g,96);
    for(unsigned u=0;u<96;++u) for(unsigned j=0;j<12;++j)
        g.Graph[u].add_base_graph_neighbors((u+1+rng()%95)%96);
    auto marks=ABMarks::layout(g); marks.top_l=20; marks.queries=10;
    marks.flags.assign(marks.offsets.back(),7);
    const auto temp=std::filesystem::current_path()/("ab_test_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".abm");
    marks.save(temp.string());
    const auto loaded=ABMarks::load(temp.string(),ABMarks::layout(g));
    check(loaded.flags==marks.flags && loaded.offsets==marks.offsets,"AB marks round trip");
    auto wrong=ABMarks::layout(g); ++wrong.fingerprint;
    bool rejected=false;
    try { ABMarks::load(temp.string(),wrong); } catch(const std::runtime_error&) { rejected=true; }
    check(rejected,"Reject different graph fingerprint");
    { std::ofstream corrupt(temp,std::ios::binary|std::ios::app); corrupt.put('x'); }
    rejected=false;
    try { ABMarks::load(temp.string(),ABMarks::layout(g)); } catch(const std::runtime_error&) { rejected=true; }
    check(rejected,"Reject wrong marks length");
    std::filesystem::remove(temp);
    g.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(std::numeric_limits<double>::max(),0);
    for(unsigned trial=0;trial<12;++trial) {
        float query[16]={}; query[0]=float(rng()%100)*.1f;
        for(size_t ef:{10,30}) {
            size_t ref_n=0,ab_n=0,plain_n=0;
            auto ref=g.searchKnnCurMidFullClean<true>(query,10,ef,ref_n);
            auto ab=g.searchKnnCurMidFullClean<true,true>(query,10,ef,ab_n,nullptr,&marks,ABMode::A);
            auto plain=g.searchKnnCurMidFullClean<false,true>(query,10,ef,plain_n,nullptr,&marks,ABMode::B);
            check(ref==ab && ab==plain && ref_n==ab_n && plain_n==0,"All marks reproduce FULL search");
            auto empty=marks; std::fill(empty.flags.begin(),empty.flags.end(),0);
            size_t beam_n=0,none_n=0;
            auto beam=g.searchKnnBeamClean<true>(query,10,ef,beam_n);
            auto none=g.searchKnnCurMidFullClean<true,true>(query,10,ef,none_n,nullptr,&empty,ABMode::A);
            check(beam==none && beam_n==none_n,"No marks reproduce beam search");
        }
    }
    // A real rejected mid followed by 20 fresh NN: no hidden NN truncation.
    HNSW_NGFix<float> full(L2_float,16,24,16,48); initialize(full,24);
    for(unsigned u=0;u<24;++u) {
        float x[16]={}; x[0]=u==0?5.f:(u==1?1.f:(u==2?2.f:.5f+u*.001f)); full.SetData(u,x);
    }
    full.Graph[0].add_base_graph_neighbors(1); full.Graph[0].add_base_graph_neighbors(2);
    for(unsigned u=3;u<23;++u) full.Graph[2].add_base_graph_neighbors(u);
    auto fm=ABMarks::layout(full); fm.flags.assign(fm.offsets.back(),0); fm.flags[1]=3;
    full.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(std::numeric_limits<double>::max(),0);
    float q[16]={}; size_t ndc=0; HNSW_NGFix<float>::CurMidFullCounters c;
    full.searchKnnCurMidFullClean<true,true>(q,1,1,ndc,&c,&fm,ABMode::A);
    check(c.probes==1 && c.nn==20 && ndc==c.entry+c.mid+c.nn,"A preserves FULL NN");
    c={}; ndc=0;
    full.searchKnnCurMidFullClean<true,true>(q,1,1,ndc,&c,&fm,ABMode::B);
    check(c.probes==0 && c.nn==0 && c.blocked_mark==1,"B rejects A-only edge");
    std::cout<<"PASS A/B reachability, duplicates, counters, FULL-NN equivalence and gating\n";
}
