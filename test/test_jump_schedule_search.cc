#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool value) { if(!value) throw std::runtime_error("search schedule check failed"); }
int main() {
    HNSW_NGFix<float> graph(L2_float,16,7,16,48);
    const float coordinates[] = {5,1,3,4,.2f,0,4};
    for(unsigned i=0;i<7;++i) {
        float vector[16]={}; vector[0]=coordinates[i];
        graph.SetData(i,vector); graph.vecdata[i*(16*sizeof(float)+1)]=0;
    }
    graph.n=7;
    // Adjacency first chooses the worse bridge; gap first finds the exact NN.
    graph.Graph[0].add_base_graph_neighbors(1);
    graph.Graph[0].add_base_graph_neighbors(3);
    graph.Graph[0].add_base_graph_neighbors(2);
    graph.Graph[3].add_base_graph_neighbors(4);
    graph.Graph[2].add_base_graph_neighbors(4);
    graph.Graph[2].add_base_graph_neighbors(5);
    graph.Graph[1].add_base_graph_neighbors(6);
    graph.Graph[6].add_base_graph_neighbors(5);
    float query[16]={};
    auto run = [&](BoundTwoHopGate gate, unsigned expected) {
        graph.bound_twohop_gate=gate;
        HNSW_NGFix<float>::TwoHopSearchStats stats;
        size_t ndc=0;
        auto result=graph.searchKnnBoundTwoHop(query,1,1,ndc,&stats);
        check(result.size()==1 && result[0].second==expected);
        check(ndc==stats.entry_distance_cmps+stats.mid_distance_cmps+stats.nn_distance_cmps);
        const auto& s=stats.gate;
        check(s.rejected_mids==stats.twohop_mid_scan_count+s.blocked_hop+s.blocked_cur+
            s.blocked_gap+s.blocked_detour+s.blocked_edge+s.blocked_budget+s.blocked_policy+
            s.blocked_selection+s.blocked_trigger+s.blocked_batch_budget);
        check(stats.twohop_mid_scan_count==s.probed_base_mids+s.probed_ngfix_mids+s.probed_rfix_mids);
        if(gate.max_nn_per_query) check(stats.nn_distance_cmps<=gate.max_nn_per_query);
        if(gate.max_nn_per_mid) check(stats.nn_distance_cmps<=stats.twohop_mid_scan_count*gate.max_nn_per_mid);
        if(gate.max_nn_per_cur) check(stats.nn_distance_cmps<=stats.cur_pop_count*gate.max_nn_per_cur);
        return stats;
    };
    BoundTwoHopGate gate;
    run(gate,5);
    gate.schedule=JumpSchedule::AFTER_ONEHOP; run(gate,5);
    gate.max_mids_per_cur=1; run(gate,4);
    gate.selection=JumpMidSelection::GAP; run(gate,5);
    gate.max_nn_per_query=1; run(gate,4);
    gate.max_nn_per_query=0; gate.max_nn_per_mid=1; gate.max_nn_per_cur=1; run(gate,4);
    gate.max_nn_per_mid=0; gate.max_nn_per_cur=0;
    gate.trigger=JumpTrigger::STALL; gate.stall_window=8;
    auto stalled=run(gate,1); check(stalled.gate.blocked_trigger>0);
    gate.trigger=JumpTrigger::STALL_FRONTIER;
    auto rescue=run(gate,5); check(rescue.gate.frontier_fires==1);
    // Visited pool reuse must not carry per-query scoring/trigger state.
    run(gate,5);
    std::cout << "Toy graph: selection, live bound, fresh-distance budgets, frontier rescue and accounting passed.\n";
}
