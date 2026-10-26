#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool x) { if(!x) throw std::runtime_error("Selective prefix equivalence/accounting failed"); }
static void compare(HNSW_NGFix<float>& graph,float* query,size_t k,size_t ef) {
    size_t a=0,b=0,unused=0;
    HNSW_NGFix<float>::TwoHopSearchStats stats;
    HNSW_NGFix<float>::SelectivePrefixCounters counts;
    auto reference=graph.searchKnnBoundTwoHop(query,k,ef,a,&stats);
    auto counted=graph.searchKnnSelectivePrefixClean<true>(query,k,ef,b,&counts);
    auto plain=graph.searchKnnSelectivePrefixClean<false>(query,k,ef,unused);
    check(reference==counted && counted==plain && a==b && unused==0);
    check(b==counts.entry+counts.mid+counts.nn && counts.entry==1);
    check(counts.nn==stats.nn_distance_cmps && counts.mid==stats.mid_distance_cmps);
    check(counts.probes==stats.twohop_mid_scan_count && counts.admitted_nn==stats.jump_candidate_count);
    check(counts.successful_probes==stats.jump_mid_count);
    check(counts.rejected==counts.probes+counts.blocked_cur+counts.blocked_gap+
        counts.blocked_budget+counts.blocked_selection);
    check(counts.nn<=graph.bound_twohop_gate.max_nn_per_query);
    check(counts.nn<=counts.probes*graph.bound_twohop_gate.nn_limit);
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
                for(size_t cap:{1,2}) for(size_t limit:{1,4,8}) {
                    graph.bound_twohop_gate=HNSW_NGFix<float>::selectivePrefixProfile(gap,margin,cap,limit,7);
                    for(auto& value:query) value=number(random);
                    compare(graph,query.data(),10,10);
                    compare(graph,query.data(),10,24);
                    compare(graph,query.data(),10,24); // visited pool reuse
                    compare(graph,query.data(),100,100);
                }
    }
    // One-hop must score node 3 before any probe can mark it visited. Prefix
    // limit 1 must leave node 4 unvisited so normal expansion of 3 discovers it.
    HNSW_NGFix<float> toy(L2_float,16,6,16,48);
    for(unsigned id=0;id<6;++id) {
        const float position[]={5,1,3,.8f,0,4};float x[16]={};x[0]=position[id];
        toy.SetData(id,x);toy.vecdata[id*(16*sizeof(float)+1)]=0;
    }
    toy.n=6;
    toy.Graph[0].add_base_graph_neighbors(1);toy.Graph[0].add_base_graph_neighbors(2);
    toy.Graph[0].add_base_graph_neighbors(3);toy.Graph[2].add_base_graph_neighbors(3);
    toy.Graph[2].add_base_graph_neighbors(5);
    toy.Graph[2].add_base_graph_neighbors(4);toy.Graph[3].add_base_graph_neighbors(4);
    toy.bound_twohop_gate=HNSW_NGFix<float>::selectivePrefixProfile(100,0,1,1,1);
    float query[16]={};compare(toy,query,1,1);
    size_t ndc=0;auto result=toy.searchKnnSelectivePrefixClean<true>(query,1,1,ndc);
    check(result.front().second==4);
    // First probe lowers bound from 1 to .25. Recheck must block the second
    // mid using the NEW cur slack, or (with cur gate off) the NEW mid gap.
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
        live.bound_twohop_gate=HNSW_NGFix<float>::selectivePrefixProfile(.04,margin,2,1,4);
        compare(live,query,2,2);
        HNSW_NGFix<float>::SelectivePrefixCounters c;ndc=0;
        live.searchKnnSelectivePrefixClean<true>(query,2,2,ndc,&c);
        check(c.probes==1 && c.successful_probes==1);
        check(margin>0 ? c.blocked_cur==1 : c.blocked_gap==1);
    }
    std::cout<<"Selective prefix: results/NDC/probes/admissions match reference; live gates, duplicates, deletion, budgets and pool reuse passed.\n";
}
