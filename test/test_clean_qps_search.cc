#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool value) { if(!value) throw std::runtime_error("Clean search equivalence failed."); }
static void compare(HNSW_NGFix<float>& graph,float* query,size_t k,size_t ef,bool must_exhaust=false) {
    graph.bound_twohop_gate=HNSW_NGFix<float>::clean32Profile();
    size_t legacy_ndc=0,beam_ndc=0,ignored=0;
    auto legacy=graph.searchKnn(query,k,ef,legacy_ndc);
    auto beam=graph.searchKnnBeamClean<true>(query,k,ef,beam_ndc);
    auto plain_beam=graph.searchKnnBeamClean<false>(query,k,ef,ignored);
    HNSW_NGFix<float>::TwoHopSearchStats beam_stats;
    size_t observed_ndc=0;
    auto observed=graph.searchKnnWithStats(query,k,ef,observed_ndc,beam_stats);
    check(legacy==beam && beam==plain_beam && beam==observed);
    check(beam_ndc==legacy_ndc+1 && beam_ndc==observed_ndc && ignored==0);
    HNSW_NGFix<float>::TwoHopSearchStats stats;
    size_t reference_ndc=0,clean_ndc=0;
    auto reference=graph.searchKnnBoundTwoHop(query,k,ef,reference_ndc,&stats);
    auto clean=graph.searchKnnCheap32Clean<true>(query,k,ef,clean_ndc);
    auto plain=graph.searchKnnCheap32Clean<false>(query,k,ef,ignored);
    check(reference==clean && reference==plain && reference_ndc==clean_ndc && ignored==0);
    if(must_exhaust) check(stats.nn_distance_cmps==4000);
}
int main() {
    std::mt19937 random(20261008);
    std::uniform_real_distribution<float> number(-1,1);
    for(Metric metric:{IP_float,L2_float}) {
        HNSW_NGFix<float> graph(metric,64,256,16,48);
        std::vector<float> x(64),query(64);
        auto vector=[&]() { float norm=0; for(auto& v:x) { v=number(random); norm+=v*v; }
            if(metric==IP_float) for(auto& v:x) v/=std::sqrt(norm); };
        for(unsigned id=0;id<256;++id) {
            vector(); graph.SetData(id,x.data()); graph.vecdata[id*(64*sizeof(float)+1)]=0;
            for(int j=0;j<28;++j) graph.Graph[id].add_base_graph_neighbors((id+1+random()%255)%256);
            graph.Graph[id].add_base_graph_neighbors((id+1)%256);
            graph.Graph[id].add_base_graph_neighbors((id+1)%256); // duplicate targets
            graph.Graph[id].add_ngfix_neighbors((id+17)%256,EH_INF,48);
            graph.Graph[id].add_ngfix_neighbors((id+33)%256,3,48);
        }
        graph.n=256;
        graph.vecdata[7*(64*sizeof(float)+1)]=1; // deletion semantics
        for(int trial=0;trial<8;++trial) {
            vector(); query=x;
            for(size_t ef:{10,20,40,80}) compare(graph,query.data(),10,ef);
        }
    }
    // Deliberately consume >4000 possible fresh NN scores. Rejected bridges
    // and their NN are near the bound but never admitted. This checks budget
    // cutoff and physical score order, not just final Recall on a small graph.
    constexpr size_t parents=400,total=1+parents+2*parents+32*parents;
    HNSW_NGFix<float> graph(L2_float,32,total,16,48);
    for(unsigned id=0;id<total;++id) {
        float x[32]={};
        x[0]=id==0 ? 0 : id<=parents ? 1 : id<=3*parents ? std::sqrt(1.01f) : std::sqrt(1.02f);
        graph.SetData(id,x); graph.vecdata[id*(32*sizeof(float)+1)]=0;
    }
    graph.n=total;
    for(unsigned parent=1;parent<=parents;++parent) {
        graph.Graph[0].add_base_graph_neighbors(parent);
        for(unsigned bridge=0;bridge<2;++bridge) {
            const unsigned mid=1+parents+2*(parent-1)+bridge;
            graph.Graph[parent].add_base_graph_neighbors(mid);
            for(unsigned j=0;j<16;++j)
                graph.Graph[mid].add_base_graph_neighbors(1+3*parents+32*(parent-1)+16*bridge+j);
        }
    }
    float query[32]={};
    compare(graph,query,10,100);
    compare(graph,query,10,400,true);
    compare(graph,query,10,400,true); // pool reuse
    std::cout << "Clean beam/cheap32: exact results and NDC match; duplicate/type/deletion/tie/budget/pool cases passed.\n";
}
