#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool x) { if(!x) throw std::runtime_error("Cur/mid full NN equivalence/accounting failed"); }
static void compare(HNSW_NGFix<float>& graph,float* query,size_t k,size_t ef) {
    size_t a=0,b=0,unused=0;
    HNSW_NGFix<float>::TwoHopSearchStats stats;
    HNSW_NGFix<float>::CurMidFullCounters counts;
    auto reference=graph.searchKnnBoundTwoHop(query,k,ef,a,&stats);
    auto counted=graph.searchKnnCurMidFullClean<true>(query,k,ef,b,&counts);
    auto plain=graph.searchKnnCurMidFullClean<false>(query,k,ef,unused);
    check(reference==counted && counted==plain && a==b && unused==0);
    check(b==counts.entry+counts.mid+counts.nn && counts.entry==1);
    check(counts.nn==stats.nn_distance_cmps && counts.mid==stats.mid_distance_cmps);
    check(counts.probes==stats.twohop_mid_scan_count && counts.admitted_nn==stats.jump_candidate_count);
    check(counts.successful_probes==stats.jump_mid_count);
    check(counts.rejected==counts.probes+counts.blocked_cur+counts.blocked_gap+
        counts.blocked_ratio+counts.blocked_selection);
    check(graph.bound_twohop_gate.max_nn_per_query==0);
    check(graph.bound_twohop_gate.max_nn_per_mid==0 && graph.bound_twohop_gate.max_nn_per_cur==0);
    check(graph.bound_twohop_gate.nn_selection==JumpNNSelection::FULL);
    if(graph.bound_twohop_gate.max_mids_per_cur)
        check(counts.probes<=counts.expanded*graph.bound_twohop_gate.max_mids_per_cur);
}
int main() {
    std::mt19937 random(20261015);
    std::uniform_real_distribution<float> number(-1,1);
    for(Metric metric:{IP_float,L2_float}) {
        HNSW_NGFix<float> graph(metric,32,192,16,48);
        std::vector<float> x(32),query(32);
        for(unsigned id=0;id<192;++id) {
            for(auto& value:x) value=number(random);
            graph.SetData(id,x.data());graph.vecdata[id*(32*sizeof(float)+1)]=0;
            for(int j=0;j<20;++j) graph.Graph[id].add_base_graph_neighbors((id+1+random()%191)%192);
            graph.Graph[id].add_base_graph_neighbors((id+1)%192);
            graph.Graph[id].add_base_graph_neighbors((id+1)%192);
        }
        graph.n=192;graph.vecdata[7*(32*sizeof(float)+1)]=1;
        for(double gap:{0.0,0.0266,0.06,std::numeric_limits<double>::max()})
            for(double margin:{0.0,0.0127,0.026})
                for(size_t cap:{0,1,2}) for(double ratio:{0.0,1.0,2.0,std::numeric_limits<double>::max()}) {
                    graph.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(gap,margin,ratio,cap);
                    for(auto& value:query) value=number(random);
                    compare(graph,query.data(),10,10);
                    compare(graph,query.data(),10,24);
                    compare(graph,query.data(),10,24); // visited pool reuse
                    compare(graph,query.data(),100,100);
                }
    }
    // This probe must score 20 NN: neither the former prefix=8 nor a per-query
    // cap may silently survive in this kernel.
    HNSW_NGFix<float> all(L2_float,16,24,16,48);
    for(unsigned id=0;id<24;++id) {
        float x[16]={};x[0]=id==0?5.f:(id==1?1.f:(id==2?2.f:.5f+id*.001f));
        all.SetData(id,x);all.vecdata[id*(16*sizeof(float)+1)]=0;
    }
    all.n=24;all.Graph[0].add_base_graph_neighbors(1);all.Graph[0].add_base_graph_neighbors(2);
    for(unsigned id=3;id<23;++id) all.Graph[2].add_base_graph_neighbors(id);
    all.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(std::numeric_limits<double>::max(),0);
    float zero[16]={};compare(all,zero,1,1);
    size_t ndc=0;HNSW_NGFix<float>::CurMidFullCounters full;
    all.searchKnnCurMidFullClean<true>(zero,1,1,ndc,&full);
    check(full.probes==1 && full.nn==20 && full.scanned_nn==20);
    // First probe tightens the live bound; the second mid must be rechecked.
    HNSW_NGFix<float> live(L2_float,16,6,16,48);
    const float positions[]={.5f,1,std::sqrt(1.02f),std::sqrt(1.03f),.4f,.3f};
    for(unsigned id=0;id<6;++id) {
        float x[16]={};x[0]=positions[id];live.SetData(id,x);
        live.vecdata[id*(16*sizeof(float)+1)]=0;
    }
    live.n=6;
    for(unsigned id:{1,2,3}) live.Graph[0].add_base_graph_neighbors(id);
    live.Graph[2].add_base_graph_neighbors(4);live.Graph[3].add_base_graph_neighbors(5);
    for(double margin:{0.0,.01}) {
        live.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(.04,margin);
        compare(live,zero,2,2);
        HNSW_NGFix<float>::CurMidFullCounters c;ndc=0;
        live.searchKnnCurMidFullClean<true>(zero,2,2,ndc,&c);
        check(c.probes==1 && (margin>0?c.blocked_cur==1:c.blocked_gap==1));
    }
    std::cout<<"Cur/mid full NN: reference results/NDC, live gates, ratio, deletion, duplicates and visited reuse passed.\n";
}
