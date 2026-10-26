#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool v) { if(!v) throw std::runtime_error("NN search check failed"); }
int main() {
    HNSW_NGFix<float> graph(L2_float,16,6,16,48);
    float positions[]={5,1,3,4,.2f,0};
    for(unsigned i=0;i<6;++i) {
        float x[16]={}; x[0]=positions[i]; graph.SetData(i,x);
        graph.vecdata[i*(16*sizeof(float)+1)]=0;
    }
    graph.n=6; graph.Graph[0].add_base_graph_neighbors(1);
    graph.Graph[0].add_base_graph_neighbors(3); graph.Graph[0].add_base_graph_neighbors(2);
    graph.Graph[2].add_base_graph_neighbors(4); graph.Graph[2].add_base_graph_neighbors(5);
    BoundTwoHopGate gate; gate.schedule=JumpSchedule::AFTER_ONEHOP;
    gate.selection=JumpMidSelection::GAP; gate.max_mids_per_cur=1;
    gate.nn_limit=1; gate.nn_cheap_dims=1;
    float query[16]={};
    auto run=[&](JumpNNSelection mode,bool trace,unsigned expected) {
        gate.nn_selection=mode; graph.bound_twohop_gate=gate;
        HNSW_NGFix<float>::TwoHopSearchStats stats;
        stats.collect_nn_events=trace; stats.collect_jump_events=trace; stats.collect_node_ids=trace;
        size_t ndc=0; auto result=graph.searchKnnBoundTwoHop(query,1,1,ndc,&stats);
        check(result.size()==1 && result[0].second==expected);
        check(ndc==stats.entry_distance_cmps+stats.mid_distance_cmps+stats.nn_distance_cmps);
        if(mode!=JumpNNSelection::FULL) check(stats.nn_distance_cmps<=stats.twohop_mid_scan_count);
        if(trace) {
            size_t scored=0; for(const auto& ev:stats.bound_nn_events) scored+=ev.scored;
            check(scored==stats.nn_distance_cmps);
        }
        return std::make_pair(ndc,stats);
    };
    auto full=run(JumpNNSelection::FULL,false,5);
    auto shadow=run(JumpNNSelection::FULL,true,5);
    check(full.first==shadow.first && full.second.nn_distance_cmps==shadow.second.nn_distance_cmps);
    check(shadow.second.gate.nn_shadow_scores==2);
    check(shadow.second.bound_nn_events.size()==2);
    for(const auto& ev:shadow.second.bound_nn_events) check(ev.scored && !ev.censored);
    auto prefix=run(JumpNNSelection::PREFIX,true,4);
    check(prefix.second.nn_distance_cmps==1 && prefix.second.bound_nn_events.size()==2);
    auto ranked=run(JumpNNSelection::CHEAP,false,5);
    check(ranked.second.gate.nn_cheap_scores==2 && ranked.second.gate.nn_cheap_products==2);
    gate.max_nn_per_query=1;
    auto capped=run(JumpNNSelection::FULL,true,4);
    check(capped.second.bound_nn_events.front().censored);
    gate.max_nn_per_query=0;
    // Skipped NN 5 must remain fresh so ordinary one-hop expansion from 4
    // can still reach it after a prefix-only jump.
    graph.Graph[4].add_base_graph_neighbors(5);
    auto rescued=run(JumpNNSelection::PREFIX,false,5);
    check(rescued.second.nn_distance_cmps==1);
    std::cout << "NN search: shadow invariance, budget censoring, cheap costs and skipped-node rescue passed.\n";
}
