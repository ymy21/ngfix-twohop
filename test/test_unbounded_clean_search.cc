#include "ngfixlib/graph/hnsw_ngfix.h"
#include <iostream>
#include <random>
using namespace ngfixlib;
int main() {
    try {
        std::mt19937 random(20261013);
        std::uniform_real_distribution<float> uniform(-1,1);
        for(Metric metric:{IP_float,L2_float}) for(bool repaired:{false,true}) {
            HNSW_NGFix<float> graph(metric,64,256,16,48);
            std::vector<float> x(64),query(64);
            for(unsigned id=0;id<256;++id) {
                for(auto& v:x) v=uniform(random);
                graph.SetData(id,x.data()); graph.vecdata[id*graph.size_per_element]=0;
                for(int j=0;j<24;++j)
                    graph.Graph[id].add_base_graph_neighbors((id+1+random()%255)%256);
                graph.Graph[id].add_base_graph_neighbors((id+1)%256);
                graph.Graph[id].add_base_graph_neighbors((id+1)%256);
                if(repaired) {
                    graph.Graph[id].add_ngfix_neighbors((id+17)%256,EH_INF,48);
                    graph.Graph[id].add_ngfix_neighbors((id+33)%256,201,48);
                }
            }
            graph.n=256;
            graph.vecdata[7*graph.size_per_element]=1;
            graph.bound_twohop_gate=BoundTwoHopGate{};
            for(int trial=0;trial<12;++trial) {
                for(auto& v:query) v=uniform(random);
                // Includes ef<K hop-limit behavior, ties/duplicates and pool reuse.
                for(size_t k:{10,100}) for(size_t ef:{5,10,20,40,80,100,200,256}) {
                    size_t r=0,c=0,ignored=0;
                    HNSW_NGFix<float>::TwoHopSearchStats stats;
                    auto reference=graph.searchKnnBoundTwoHop(query.data(),k,ef,r,&stats);
                    auto counted=graph.searchKnnBoundUnboundedClean<true>(query.data(),k,ef,c);
                    auto plain=graph.searchKnnBoundUnboundedClean<false>(query.data(),k,ef,ignored);
                    if(reference!=counted || reference!=plain || r!=c || ignored!=0)
                        throw std::runtime_error("Unbounded clean/reference mismatch");
                    if(trial==0 && ef==100) {
                        HNSW_NGFix<float>::TwoHopSearchStats traced;
                        traced.collect_search_state=true;traced.collect_jump_events=true;
                        size_t traced_ndc=0;
                        auto observed=graph.searchKnnBoundTwoHop(query.data(),k,ef,traced_ndc,&traced);
                        if(observed!=reference || traced_ndc!=r)
                            throw std::runtime_error("Search state instrumentation changed results/NDC");
                        for(const auto& probe:traced.bound_mid_probe_events) {
                            const auto& state=probe.state;
                            if(state.ndc!=1+state.ndc_mid+state.ndc_nn || state.step!=probe.hop ||
                               state.topk_size>k || state.topk_size>state.beam_size || state.beam.variance<0)
                                throw std::runtime_error("Invalid pre-probe snapshot");
                        }
                    }
                }
            }
        }
        std::cout << "Unbounded clean/reference results and NDC match.\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
