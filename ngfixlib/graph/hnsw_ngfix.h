#pragma once

#include <memory>
#include <type_traits>
#include <atomic>
#include <shared_mutex>
#include <bitset>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <map>
#include <array>
#include <cmath>
#include <functional>
#include <queue>
#include <unordered_map>
#include <type_traits>
#include <fstream>
#include "node.h"
#include "ab_marks.h"
#include "bound_twohop_gate.h"
#include "jump_policy.h"
#include "ngfixlib/utils/search_list.h"
#include "ngfixlib/utils/visited_list.h"
#include "ngfixlib/metric/l2.h"
#include "ngfixlib/metric/ip.h"
#include "ngfixlib/metric/rabitq.h"
#include "rabitqlib/quantization/rabitq.hpp"

static const size_t MAX_Nq = 200;
static const size_t MAX_S = 200;

namespace ngfixlib {

enum Metric{
    L2_float = 1,
    IP_float = 2,
    L2_uint8 = 3,
    IP_uint8 = 4,
    L2_RaBitQ = 5,
    IP_RaBitQ = 6,
    L2_RaBitQ_b8_q8 = 7,
    IP_RaBitQ_b8_q8 = 8,
    IP_RaBitQ_b8_q32 = 9,
};

template<typename T>
class HNSW_NGFix
{
private:
    // lock when updating neighbors 
    std::vector<std::shared_mutex> node_locks;
    size_t M0 = 0; // out-degree of base layer  
    size_t M = 0;  // used for insertion
    size_t MEX = 0;

public:
    std::vector<node> Graph;
    char* vecdata;
    Space<T>* space;
    Space<T>* query_space; // used when q_bits != b_bits
    size_t dim;
    size_t entry_point = 0;
    size_t max_elements;
    std::atomic<size_t> n = 0;
    std::shared_ptr<VisitedListPool> visited_list_pool_{nullptr};
    
    std::shared_mutex delete_lock;
    std::unordered_set<id_t> delete_ids;

    // ===== RoarGraph-style search modification =====
    // false: original NGFix search
    // true : enable cur -> mid -> nn traversal
    bool enable_twohop_search = false;
    bool enable_history_twohop = false;
    float history_threshold = 0.05f;
    std::vector<float> history_frequency;
    BoundTwoHopGate bound_twohop_gate;
    JumpPolicy jump_policy;

    // Edge types used by search/statistics.
    // Storage layout:
    //   repair area [NGFix + RFix] | HNSW base area
    // Repair-area semantic split:
    //   eh == MAX_S + 1 : RFix edge (RFix() writes this marker explicitly)
    //   otherwise       : NGFix edge (finite hardness or EH_INF)
    enum class SearchEdgeType : uint8_t {
        HNSW_BASE = 0,
        NGFIX = 1,
        RFIX = 2
    };

    // Exact NGFix EH histogram: indices [0, MAX_S] are finite EH values,
    // index MAX_S+1 is EH_INF. RFix marker (MAX_S+1 in storage) is
    // classified before histogramming and therefore never enters this array.
    static constexpr size_t EH_HIST_INF_INDEX = MAX_S + 1;
    static constexpr size_t EH_HIST_SIZE = MAX_S + 2;

    static size_t ehHistIndex(uint16_t eh) {
        return eh == EH_INF ? EH_HIST_INF_INDEX
                            : std::min<size_t>(static_cast<size_t>(eh), MAX_S);
    }

    struct NodeLocalStructure {
        size_t total_degree = 0;
        size_t hnsw_degree = 0;
        size_t ngfix_degree = 0;
        size_t rfix_degree = 0;
        size_t repair_degree = 0;
        size_t ngfix_finite_degree = 0;
        size_t ngfix_eh_inf_degree = 0;
        double repair_ratio = 0.0;
    };

    struct PairLocalStructure {
        float data_dist = std::numeric_limits<float>::infinity();
        int a_to_b_edge_type = -1;
        int b_to_a_edge_type = -1;
        size_t common_neighbors = 0;
        size_t union_neighbors = 0;
        double neighbor_jaccard = 0.0;
    };

    struct GraphEdgeStats {
        // All counts are directed-edge counts.
        size_t total_edges = 0;
        size_t hnsw_base_edges = 0;
        size_t repair_area_edges = 0;
        size_t ngfix_edges = 0;
        size_t rfix_edges = 0;

        // Extra NGFix-only EH breakdown. RFix(marker=MAX_S+1) is excluded.
        size_t ngfix_eh_inf_edges = 0;
        size_t ngfix_finite_eh_edges = 0;
        std::array<size_t, EH_HIST_SIZE> ngfix_eh_hist{};
    };



    // Structural audit for the repair area in an already-built index.
    // This is a final-graph saturation proxy; it does NOT reconstruct whether
    // a source was full at the exact moment each edge was inserted.
    struct RepairDegreeStats {
        size_t node_count = 0;
        size_t nodes_with_repair = 0;
        size_t nodes_without_repair = 0;
        size_t nodes_below_mex = 0;
        size_t nodes_at_mex = 0;
        size_t nodes_above_mex = 0;
        size_t total_repair_edges = 0;
        size_t max_repair_degree = 0;
        size_t duplicate_repair_entries = 0;

        size_t eh_0_20_edges = 0;
        size_t eh_21_100_edges = 0;
        size_t eh_101_200_edges = 0;
        size_t eh_inf_edges = 0;
        size_t rfix_edges = 0;

        size_t eh_0_20_from_below_mex = 0;
        size_t eh_0_20_from_at_mex = 0;
        size_t eh_21_100_from_below_mex = 0;
        size_t eh_21_100_from_at_mex = 0;
        size_t eh_101_200_from_below_mex = 0;
        size_t eh_101_200_from_at_mex = 0;
        size_t eh_inf_from_below_mex = 0;
        size_t eh_inf_from_at_mex = 0;
        size_t rfix_from_below_mex = 0;
        size_t rfix_from_at_mex = 0;

        std::vector<size_t> degree_hist;
    };

    struct EdgeUsageStats {
        // Stage 1: adjacency entries scanned, including already-visited targets.
        size_t examined_total = 0;
        size_t examined_hnsw_base = 0;
        size_t examined_ngfix = 0;
        size_t examined_rfix = 0;
        size_t examined_ngfix_eh_inf = 0;
        size_t examined_ngfix_finite_eh = 0;
        std::array<size_t, EH_HIST_SIZE> examined_ngfix_eh_hist{};

        // Stage 2: target was fresh, so an exact query-to-target distance was computed.
        // This is the old code's actual "enqueued" semantics; renamed for correctness.
        size_t scored_total = 0;
        size_t scored_hnsw_base = 0;
        size_t scored_ngfix = 0;
        size_t scored_rfix = 0;
        size_t scored_ngfix_eh_inf = 0;
        size_t scored_ngfix_finite_eh = 0;
        std::array<size_t, EH_HIST_SIZE> scored_ngfix_eh_hist{};

        // Stage 3: after scoring, Search_QuadHeap's admission predicate passed,
        // so the target was inserted into candidate_set (and into top_candidates
        // as well when it is not deleted).
        size_t admitted_total = 0;
        size_t admitted_hnsw_base = 0;
        size_t admitted_ngfix = 0;
        size_t admitted_rfix = 0;
        size_t admitted_ngfix_eh_inf = 0;
        size_t admitted_ngfix_finite_eh = 0;
        std::array<size_t, EH_HIST_SIZE> admitted_ngfix_eh_hist{};

        // Stage 4: an admitted target was later popped and actually expanded.
        // Counts are attributed to the edge through which that target was first discovered.
        size_t expanded_from_total = 0;
        size_t expanded_from_hnsw_base = 0;
        size_t expanded_from_ngfix = 0;
        size_t expanded_from_rfix = 0;
        size_t expanded_from_ngfix_eh_inf = 0;
        size_t expanded_from_ngfix_finite_eh = 0;
        std::array<size_t, EH_HIST_SIZE> expanded_from_ngfix_eh_hist{};

        // Legacy counters kept only because the experimental two-hop code calls
        // recordEnqueuedEdge(). They mean "push attempted after a fresh distance"
        // and MUST NOT be interpreted as true heap admission. Baseline measurement
        // does not use these fields.
        size_t enqueued_total = 0;
        size_t enqueued_hnsw_base = 0;
        size_t enqueued_ngfix = 0;
        size_t enqueued_rfix = 0;
        size_t enqueued_ngfix_eh_inf = 0;
        size_t enqueued_ngfix_finite_eh = 0;
    };

    struct EdgePairStats {
        size_t total = 0;
        size_t hnsw_hnsw = 0;
        size_t hnsw_ngfix = 0;
        size_t hnsw_rfix = 0;
        size_t ngfix_hnsw = 0;
        size_t ngfix_ngfix = 0;
        size_t ngfix_rfix = 0;
        size_t rfix_hnsw = 0;
        size_t rfix_ngfix = 0;
        size_t rfix_rfix = 0;
    };

    #include "bound_search_snapshot.h"

    struct TwoHopSearchStats {
        BoundGateStats gate;
        // Optional event-level instrumentation. Disabled by default so normal
        // profiling/search runs do not pay the unordered-set/vector overhead.
        bool collect_node_ids = false;
        bool collect_jump_events = false;
        bool collect_nn_events = false;
        bool collect_search_state = false;

        // Search-time query-to-node distance computations only.
        // Includes the entry point and excludes evaluation-only computations
        // (recall / rderr / ground-truth evaluation).
        // distance_cmps = entry_distance_cmps + mid_distance_cmps + nn_distance_cmps.
        size_t distance_cmps = 0;
        size_t entry_distance_cmps = 0;
        size_t mid_distance_cmps = 0;
        size_t nn_distance_cmps = 0;
        // Fresh NN distance computations are attributed by the FINAL outcome
        // of the mid that owns the second-hop scan:
        //   jump:        the mid found >=1 nn with nn_dist < cur_dist;
        //   failed_jump: the mid scanned its nn list but found no such nn.
        // Every fresh NN distance computation under that mid is assigned to
        // exactly one bucket, so their sum equals nn_distance_cmps.
        size_t nn_jump_distance_cmps = 0;
        size_t nn_nojump_distance_cmps = 0;
        size_t cache_hits = 0;

        size_t cur_pop_count = 0;
        size_t hop_limit_stop_count = 0;
        size_t onehop_mid_examined = 0;
        size_t twohop_mid_scan_count = 0;
        size_t checked_nn_count = 0;

        // Mid classification for two-hop search.
        // direct_mid:       mid_dist <= cur_dist, so no second-hop scan is needed.
        // jump_mid:         mid_dist > cur_dist and second-hop scan finds a jump.
        // failed_jump_mid:  mid_dist > cur_dist and second-hop scan finds no jump.
        // skipped_mid:      mid was already ENQUEUED before this adjacency entry.
        size_t direct_mid_count = 0;
        size_t jump_mid_count = 0;
        size_t failed_jump_mid_count = 0;
        size_t skipped_mid_count = 0;
        // Event-trace-only attribution for UNIQUE first-hop nodes skipped
        // because they had already been scored during an earlier second-hop probe.
        // This quantifies how much of the apparent mid-NDC reduction is merely
        // a displacement of work from first hop to second hop.
        size_t skipped_mid_from_success_probe = 0;
        size_t skipped_mid_from_failed_probe = 0;
        size_t skipped_mid_from_admitted_jump = 0;
        size_t jump_candidate_count = 0;
        size_t mid_enqueue_count = 0;
        size_t nn_enqueue_count = 0;

        size_t history_allowed_cur_count = 0;
        size_t history_fallback_cur_count = 0;

        EdgeUsageStats first_hop_edges;
        EdgeUsageStats second_hop_edges;

        // 3x3 edge-type combinations for every scanned cur->mid->nn adjacency path.
        // This includes second-hop targets that are already visited.
        EdgePairStats twohop_examined_pairs;

        // 3x3 edge-type combinations for second-hop targets whose query
        // distance is freshly computed in bound-two-hop.  This is the clean
        // denominator for admission yield, because visited targets are removed.
        EdgePairStats twohop_scored_pairs;

        // 3x3 edge-type combinations for paths satisfying the jump condition.
        // In searchKnnTwoHopExpansion this is observational only.
        EdgePairStats jump_pairs;

        // Diagnostic provenance for bound-triggered bridge search. These fields
        // are intentionally instrumentation-only (not suitable for latency
        // claims):
        //   jump_admitted_nodes: nodes inserted directly through mid -> nn jump;
        //   bridge_root_by_node[x]: first jump node whose search subtree first
        //     discovered/admitted x. This lets evaluation distinguish direct
        //     GT jumps from non-GT routing jumps that later discover GT nodes.
        std::unordered_set<id_t> jump_admitted_nodes;
        std::unordered_map<id_t, id_t> bridge_root_by_node;

        // Parent relation between jump roots. bridge_root_by_node keeps the
        // NEAREST/most-recent jump root for unique attribution. If a later jump
        // happens inside an earlier jump's search subtree, jump_parent_root lets
        // evaluation walk the whole jump ancestry and ask whether the earlier
        // non-GT jump eventually helped reach a GT through nested jumps.
        std::unordered_map<id_t, id_t> jump_parent_root;

        // When collect_node_ids=true, record which nodes were actually scored
        // (query distance computed) and which nodes were popped/expanded. This
        // is primarily used by boundtwohopcmp to distinguish "baseline never
        // saw this jump root" from "baseline scored it but did not exploit it".
        std::unordered_set<id_t> scored_nodes;
        std::unordered_set<id_t> expanded_nodes;

        struct BoundJumpEvent {
            id_t cur = 0;
            id_t mid = 0;
            id_t nn = 0;
            size_t hop = 0;
            float cur_dist = 0.0f;
            float mid_dist = 0.0f;
            float nn_dist = 0.0f;
            float bound_at_mid_reject = 0.0f;
            float bound_before_nn_admit = 0.0f;
            SearchEdgeType first_edge_type = SearchEdgeType::HNSW_BASE;
            SearchEdgeType second_edge_type = SearchEdgeType::HNSW_BASE;
            uint16_t first_edge_eh = EH_INF;
            uint16_t second_edge_eh = EH_INF;
            size_t rank_in_fresh_nn_by_query_dist = 0;
            size_t mid_checked_nn = 0;
            size_t mid_fresh_nn_cmps = 0;
        };

        struct BoundNNEvent {
            size_t probe_id = 0, hop = 0, pool_size = 0, scored_size = 0;
            id_t cur = 0, mid = 0, nn = 0;
            int adjacency_ordinal = 0;
            unsigned first_type = 0, second_type = 0;
            uint16_t second_eh = EH_INF;
            float exact_dist = std::numeric_limits<float>::quiet_NaN();
            float cheap_score = 0;
            uint64_t random_key = 0;
            bool selected = false, scored = false, admitted = false, censored = false;
            size_t exact_rank = 0;
        };
        std::vector<BoundNNEvent> bound_nn_events;

        struct BoundMidProbeEvent {
            size_t probe_id = 0;
            BoundSearchSnapshot state;
            id_t cur = 0;
            id_t mid = 0;
            size_t hop = 0;
            float cur_dist = 0.0f;
            float mid_dist = 0.0f;
            float bound_at_reject = 0.0f;
            SearchEdgeType first_edge_type = SearchEdgeType::HNSW_BASE;
            uint16_t first_edge_eh = EH_INF;
            size_t checked_nn = 0;
            size_t fresh_nn_cmps = 0;
            size_t admitted_nn = 0;
            id_t best_scored_nn = std::numeric_limits<id_t>::max();
            float best_scored_nn_dist = std::numeric_limits<float>::infinity();
            SearchEdgeType best_scored_second_edge_type = SearchEdgeType::HNSW_BASE;
            uint16_t best_scored_second_edge_eh = EH_INF;
            id_t best_admitted_nn = std::numeric_limits<id_t>::max();
            float best_admitted_nn_dist = std::numeric_limits<float>::infinity();
        };

        // These vectors are only populated for bound-triggered two-hop when
        // collect_jump_events=true.  A mid event exists for every rejected mid
        // that is probed; a jump event exists for every second-hop nn actually
        // admitted by the live beam predicate.
        std::vector<BoundJumpEvent> bound_jump_events;
        std::vector<BoundMidProbeEvent> bound_mid_probe_events;
    };
    // ===== End modification =====

    size_t size_per_element = 0;

    HNSW_NGFix(Metric metric, size_t dimension, size_t max_elements, size_t M_ = 16, size_t MEX_ = 48)
                : node_locks(max_elements), M(M_), MEX(MEX_) {
        M0 = 2*M;
        this->dim = dimension;
        this->max_elements = max_elements;
        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag

        if constexpr (std::is_same_v<T, float>) {
            if (metric == L2_float) {
                space = new L2Space_float(dim);
                query_space = new L2Space_float(dim);
            } else if(metric == IP_float) {
                space = new IPSpace_float(dim);
                query_space = new IPSpace_float(dim);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if(metric == IP_RaBitQ_b8_q8) {
                space = new IPSpace_RaBitQ(dim, 8, 8);
                query_space = new IPSpace_RaBitQ(dim, 8, 8);
            } else if(metric == L2_RaBitQ_b8_q8) {
                space = new L2Space_RaBitQ(dim, 8, 8);
                query_space = new L2Space_RaBitQ(dim, 8, 8);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }   
        
        vecdata = new char[size_per_element*max_elements];
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
    }

    HNSW_NGFix(Metric metric, std::string path) {
        std::ifstream input(path, std::ios::binary);

        input.read((char*)&M, sizeof(M));
        input.read((char*)&M0, sizeof(M0));
        input.read((char*)&MEX, sizeof(MEX));
        input.read((char*)&n, sizeof(n));
        input.read((char*)&entry_point, sizeof(entry_point));
        input.read((char*)&dim, sizeof(dim));
        input.read((char*)&max_elements, sizeof(max_elements));
        this->size_per_element = dim*sizeof(T) + 1; // 8 bits for delete flag
        if constexpr (std::is_same_v<T, float>) {
            if (metric == L2_float) {
                space = new L2Space_float(dim);
                query_space = new L2Space_float(dim);
            } else if(metric == IP_float) {
                space = new IPSpace_float(dim);
                query_space = new IPSpace_float(dim);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if(metric == IP_RaBitQ_b8_q8) {
                space = new IPSpace_RaBitQ(dim, 8, 8);
                query_space = new IPSpace_RaBitQ(dim, 8, 8);
            } else if(metric == L2_RaBitQ_b8_q8) {
                space = new L2Space_RaBitQ(dim, 8, 8);
                query_space = new L2Space_RaBitQ(dim, 8, 8);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }

        vecdata = new char[size_per_element*max_elements];
        input.read(vecdata, max_elements*size_per_element);

        std::vector<std::shared_mutex>(this->max_elements).swap(node_locks);
        Graph.resize(this->max_elements);
        visited_list_pool_ = std::shared_ptr<VisitedListPool>(new VisitedListPool(1, this->max_elements));
        
        for(int i = 0; i < n; ++i) {
            Graph[i].LoadIndex(input);
        }
    }

    ~HNSW_NGFix() {
        delete []vecdata;
        delete space;
        delete query_space;
    }

    void set_new_metric(Metric metric) {
        delete space;
        delete query_space;
        if constexpr (std::is_same_v<T, float>) {
            if (metric == L2_float) {
                space = new L2Space_float(dim);
                query_space = new L2Space_float(dim);
            } else if(metric == IP_float) {
                space = new IPSpace_float(dim);
                query_space = new IPSpace_float(dim);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            if(metric == IP_RaBitQ_b8_q8) {
                space = new IPSpace_RaBitQ(dim, 8, 8);
                query_space = new IPSpace_RaBitQ(dim, 8, 8);
            } else if(metric == IP_RaBitQ_b8_q32) {
                space = new IPSpace_RaBitQ(dim, 8, 8);
                query_space = new IPSpace_RaBitQ(dim, 8, 32); // only using 32 bits for querying and fixing
            } else if(metric == L2_RaBitQ_b8_q8) {
                space = new L2Space_RaBitQ(dim, 8, 8);
                query_space = new L2Space_RaBitQ(dim, 8, 8);
            } else {
                throw std::runtime_error("Error: Unsupported metric type.");
            }
        } else {
            throw std::runtime_error("Error: Unsupported metric type.");
        }
    }

    void resize(size_t new_max_elements) {
        if(new_max_elements <= this->max_elements) {
            return;
        }
        
        auto new_vecdata = new char[size_per_element*new_max_elements];
        memcpy(new_vecdata, vecdata, size_per_element*max_elements);        
        delete []vecdata;
        vecdata = new_vecdata;

        this->max_elements = new_max_elements;
        std::vector<std::shared_mutex>(new_max_elements).swap(node_locks);
        Graph.resize(new_max_elements);
        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));
    }

    void StoreIndex(std::string path) {
        std::ofstream output(path, std::ios::binary);

        output.write((char*)&M, sizeof(M));
        output.write((char*)&M0, sizeof(M0));
        output.write((char*)&MEX, sizeof(MEX));
        output.write((char*)&n, sizeof(n));
        output.write((char*)&entry_point, sizeof(entry_point));
        output.write((char*)&dim, sizeof(dim));
        output.write((char*)&max_elements, sizeof(max_elements));

        output.write(vecdata, max_elements*size_per_element);
        for(int i = 0; i < n; ++i) {
            Graph[i].StoreIndex(output);
        }
    }

    T* getData(id_t u) {
        return (T*)(vecdata + u*size_per_element + 1);
    }

    void SetData(id_t u, T* data) {
        memcpy(getData(u), data, sizeof(T)*dim);
    }

    // <neighbors, out-degree>
    auto getNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp) + 1};
    }

    auto getBaseGraphNeighbors(id_t u) {
        auto tmp = Graph[u].get_neighbors();
        return std::tuple{tmp, GET_SZ((uint8_t*)tmp)-GET_NGFIX_SZ((uint8_t*)tmp), GET_NGFIX_CAPACITY((uint8_t*)tmp) + 1};
    }

    float getDist(id_t u, id_t v) {
        return space->dist_func(getData(u), getData(v));
    }

    float getDist(id_t u, T* data) {
        return space->dist_func(getData(u), data);
    }

    float getQueryDist(id_t u, T* query_data) {
        return query_space->dist_func(getData(u), query_data);
    }

    // Offline/event-trace helpers. These DO NOT contribute to search NDC and
    // should only be used for characterization after a query finishes.
    NodeLocalStructure getNodeLocalStructure(id_t u) const {
        NodeLocalStructure out;
        if(static_cast<size_t>(u) >= static_cast<size_t>(n)) return out;
        auto neighbors = Graph[u].neighbors;
        if(neighbors == nullptr) return out;
        const int sz = GET_SZ((uint8_t*)neighbors);
        const int ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        const int ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        const int st = ngfix_capacity - ngfix_sz + 1;
        for(int i = st; i < st + sz; ++i) {
            const SearchEdgeType type = getSearchEdgeType(u, i);
            ++out.total_degree;
            if(type == SearchEdgeType::HNSW_BASE) {
                ++out.hnsw_degree;
            } else if(type == SearchEdgeType::RFIX) {
                ++out.rfix_degree;
                ++out.repair_degree;
            } else {
                ++out.ngfix_degree;
                ++out.repair_degree;
                const uint16_t eh = getSearchEdgeEh(u, i);
                if(eh == EH_INF) ++out.ngfix_eh_inf_degree;
                else ++out.ngfix_finite_degree;
            }
        }
        if(out.total_degree > 0)
            out.repair_ratio = static_cast<double>(out.repair_degree) / static_cast<double>(out.total_degree);
        return out;
    }

    int getDirectedEdgeType(id_t a, id_t b) const {
        if(static_cast<size_t>(a) >= static_cast<size_t>(n) ||
           static_cast<size_t>(b) >= static_cast<size_t>(n)) return -1;
        auto neighbors = Graph[a].neighbors;
        if(neighbors == nullptr) return -1;
        const int sz = GET_SZ((uint8_t*)neighbors);
        const int ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        const int ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
        const int st = ngfix_capacity - ngfix_sz + 1;
        for(int i = st; i < st + sz; ++i) {
            if(neighbors[i] == b) return static_cast<int>(getSearchEdgeType(a, i));
        }
        return -1;
    }

    PairLocalStructure getPairLocalStructure(id_t a, id_t b) {
        PairLocalStructure out;
        if(static_cast<size_t>(a) >= static_cast<size_t>(n) ||
           static_cast<size_t>(b) >= static_cast<size_t>(n)) return out;
        out.data_dist = getDist(a, b);
        out.a_to_b_edge_type = getDirectedEdgeType(a, b);
        out.b_to_a_edge_type = getDirectedEdgeType(b, a);

        auto collect_unique_neighbors = [&](id_t u) {
            std::unordered_set<id_t> ids;
            auto neighbors = Graph[u].neighbors;
            if(neighbors == nullptr) return ids;
            const int sz = GET_SZ((uint8_t*)neighbors);
            const int ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
            const int ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            const int st = ngfix_capacity - ngfix_sz + 1;
            ids.reserve(static_cast<size_t>(sz) * 2 + 1);
            for(int i = st; i < st + sz; ++i) ids.insert(neighbors[i]);
            return ids;
        };

        auto na = collect_unique_neighbors(a);
        auto nb = collect_unique_neighbors(b);
        if(na.size() > nb.size()) std::swap(na, nb);
        size_t common = 0;
        for(const id_t x : na) if(nb.count(x)) ++common;
        out.common_neighbors = common;
        out.union_neighbors = na.size() + nb.size() - common;
        if(out.union_neighbors > 0)
            out.neighbor_jaccard = static_cast<double>(common) / static_cast<double>(out.union_neighbors);
        return out;
    }

    static constexpr uint16_t RFIX_EH_MARKER = static_cast<uint16_t>(MAX_S + 1);

    const char* searchEdgeTypeName(SearchEdgeType type) const {
        switch(type) {
            case SearchEdgeType::HNSW_BASE: return "HNSW_BASE";
            case SearchEdgeType::NGFIX: return "NGFIX";
            case SearchEdgeType::RFIX: return "RFIX";
        }
        return "UNKNOWN";
    }

    uint16_t getSearchEdgeEh(id_t u, int neighbor_array_index) const {
        auto neighbors = Graph[u].neighbors;
        const int ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
        if(neighbor_array_index > ngfix_capacity ||
           neighbor_array_index <= 0 ||
           Graph[u].ehs == nullptr) {
            return EH_INF;
        }
        return Graph[u].ehs[neighbor_array_index - 1];
    }

    SearchEdgeType getSearchEdgeType(id_t u, int neighbor_array_index) const {
        auto neighbors = Graph[u].neighbors;
        const int ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);

        // Base-graph edges are physically stored after the shared repair area.
        if(neighbor_array_index > ngfix_capacity) {
            return SearchEdgeType::HNSW_BASE;
        }

        // NGFix and RFix share the same repair-area storage. RFix() explicitly
        // writes eh=MAX_S+1; all other repair entries are classified as NGFix.
        const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
        return (eh == RFIX_EH_MARKER) ? SearchEdgeType::RFIX : SearchEdgeType::NGFIX;
    }

    void recordExaminedEdge(EdgeUsageStats& stats, id_t u, int neighbor_array_index) const {
        ++stats.examined_total;
        const SearchEdgeType type = getSearchEdgeType(u, neighbor_array_index);
        if(type == SearchEdgeType::HNSW_BASE) {
            ++stats.examined_hnsw_base;
        } else if(type == SearchEdgeType::RFIX) {
            ++stats.examined_rfix;
        } else {
            ++stats.examined_ngfix;
            const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
            ++stats.examined_ngfix_eh_hist[ehHistIndex(eh)];
            if(eh == EH_INF) ++stats.examined_ngfix_eh_inf;
            else ++stats.examined_ngfix_finite_eh;
        }
    }

    void recordScoredEdge(EdgeUsageStats& stats, id_t u, int neighbor_array_index) const {
        ++stats.scored_total;
        const SearchEdgeType type = getSearchEdgeType(u, neighbor_array_index);
        if(type == SearchEdgeType::HNSW_BASE) {
            ++stats.scored_hnsw_base;
        } else if(type == SearchEdgeType::RFIX) {
            ++stats.scored_rfix;
        } else {
            ++stats.scored_ngfix;
            const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
            ++stats.scored_ngfix_eh_hist[ehHistIndex(eh)];
            if(eh == EH_INF) ++stats.scored_ngfix_eh_inf;
            else ++stats.scored_ngfix_finite_eh;
        }
    }

    void recordAdmittedEdge(EdgeUsageStats& stats, id_t u, int neighbor_array_index) const {
        ++stats.admitted_total;
        const SearchEdgeType type = getSearchEdgeType(u, neighbor_array_index);
        if(type == SearchEdgeType::HNSW_BASE) {
            ++stats.admitted_hnsw_base;
        } else if(type == SearchEdgeType::RFIX) {
            ++stats.admitted_rfix;
        } else {
            ++stats.admitted_ngfix;
            const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
            ++stats.admitted_ngfix_eh_hist[ehHistIndex(eh)];
            if(eh == EH_INF) ++stats.admitted_ngfix_eh_inf;
            else ++stats.admitted_ngfix_finite_eh;
        }
    }

    void recordExpandedFromEdge(EdgeUsageStats& stats, id_t u, int neighbor_array_index) const {
        ++stats.expanded_from_total;
        const SearchEdgeType type = getSearchEdgeType(u, neighbor_array_index);
        if(type == SearchEdgeType::HNSW_BASE) {
            ++stats.expanded_from_hnsw_base;
        } else if(type == SearchEdgeType::RFIX) {
            ++stats.expanded_from_rfix;
        } else {
            ++stats.expanded_from_ngfix;
            const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
            ++stats.expanded_from_ngfix_eh_hist[ehHistIndex(eh)];
            if(eh == EH_INF) ++stats.expanded_from_ngfix_eh_inf;
            else ++stats.expanded_from_ngfix_finite_eh;
        }
    }

    // Legacy helper used by the old two-hop experiments. This is a push-attempt
    // counter, NOT a true admission counter. Baseline measurement uses
    // recordScoredEdge()/recordAdmittedEdge() instead.
    void recordEnqueuedEdge(EdgeUsageStats& stats, id_t u, int neighbor_array_index) const {
        ++stats.enqueued_total;
        const SearchEdgeType type = getSearchEdgeType(u, neighbor_array_index);
        if(type == SearchEdgeType::HNSW_BASE) {
            ++stats.enqueued_hnsw_base;
        } else if(type == SearchEdgeType::RFIX) {
            ++stats.enqueued_rfix;
        } else {
            ++stats.enqueued_ngfix;
            const uint16_t eh = getSearchEdgeEh(u, neighbor_array_index);
            if(eh == EH_INF) ++stats.enqueued_ngfix_eh_inf;
            else ++stats.enqueued_ngfix_finite_eh;
        }
    }

    void recordEdgePair(EdgePairStats& stats, SearchEdgeType first, SearchEdgeType second) const {
        ++stats.total;
        if(first == SearchEdgeType::HNSW_BASE) {
            if(second == SearchEdgeType::HNSW_BASE) ++stats.hnsw_hnsw;
            else if(second == SearchEdgeType::NGFIX) ++stats.hnsw_ngfix;
            else ++stats.hnsw_rfix;
        } else if(first == SearchEdgeType::NGFIX) {
            if(second == SearchEdgeType::HNSW_BASE) ++stats.ngfix_hnsw;
            else if(second == SearchEdgeType::NGFIX) ++stats.ngfix_ngfix;
            else ++stats.ngfix_rfix;
        } else {
            if(second == SearchEdgeType::HNSW_BASE) ++stats.rfix_hnsw;
            else if(second == SearchEdgeType::NGFIX) ++stats.rfix_ngfix;
            else ++stats.rfix_rfix;
        }
    }

    GraphEdgeStats getGraphEdgeStats() const {
        GraphEdgeStats stats;
        for(size_t u = 0; u < static_cast<size_t>(n); ++u) {
            auto neighbors = Graph[u].neighbors;
            const size_t sz = GET_SZ((uint8_t*)neighbors);
            const size_t repair_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            const size_t base_sz = sz - repair_sz;
            const size_t repair_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);

            stats.total_edges += sz;
            stats.hnsw_base_edges += base_sz;
            stats.repair_area_edges += repair_sz;

            if(repair_sz > 0 && Graph[u].ehs != nullptr) {
                const size_t begin = repair_capacity - repair_sz;
                for(size_t e = begin; e < repair_capacity; ++e) {
                    const uint16_t eh = Graph[u].ehs[e];
                    if(eh == RFIX_EH_MARKER) {
                        ++stats.rfix_edges;
                    } else {
                        ++stats.ngfix_edges;
                        ++stats.ngfix_eh_hist[ehHistIndex(eh)];
                        if(eh == EH_INF) ++stats.ngfix_eh_inf_edges;
                        else ++stats.ngfix_finite_eh_edges;
                    }
                }
            }
        }
        return stats;
    }


    RepairDegreeStats getRepairDegreeStats() const {
        RepairDegreeStats stats;
        stats.node_count = static_cast<size_t>(n);
        stats.degree_hist.assign(std::max<size_t>(MEX + 1, 1), 0);

        for(size_t u = 0; u < static_cast<size_t>(n); ++u) {
            auto neighbors = Graph[u].neighbors;
            const size_t repair_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            const size_t repair_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
            stats.total_repair_edges += repair_sz;
            stats.max_repair_degree = std::max(stats.max_repair_degree, repair_sz);
            if(repair_sz >= stats.degree_hist.size()) stats.degree_hist.resize(repair_sz + 1, 0);
            ++stats.degree_hist[repair_sz];

            if(repair_sz == 0) ++stats.nodes_without_repair;
            else ++stats.nodes_with_repair;
            if(repair_sz < MEX) ++stats.nodes_below_mex;
            else if(repair_sz == MEX) ++stats.nodes_at_mex;
            else ++stats.nodes_above_mex;

            if(repair_sz == 0 || Graph[u].ehs == nullptr) continue;
            const bool source_below_mex = repair_sz < MEX;
            const bool source_at_mex = repair_sz == MEX;
            const size_t begin = repair_capacity - repair_sz;
            std::unordered_set<id_t> unique_targets;
            unique_targets.reserve(repair_sz * 2 + 1);

            for(size_t e = begin; e < repair_capacity; ++e) {
                const id_t v = neighbors[e + 1];
                if(!unique_targets.insert(v).second) ++stats.duplicate_repair_entries;

                const uint16_t eh = Graph[u].ehs[e];
                if(eh == RFIX_EH_MARKER) {
                    ++stats.rfix_edges;
                    if(source_below_mex) ++stats.rfix_from_below_mex;
                    if(source_at_mex) ++stats.rfix_from_at_mex;
                } else if(eh == EH_INF) {
                    ++stats.eh_inf_edges;
                    if(source_below_mex) ++stats.eh_inf_from_below_mex;
                    if(source_at_mex) ++stats.eh_inf_from_at_mex;
                } else if(eh <= 20) {
                    ++stats.eh_0_20_edges;
                    if(source_below_mex) ++stats.eh_0_20_from_below_mex;
                    if(source_at_mex) ++stats.eh_0_20_from_at_mex;
                } else if(eh <= 100) {
                    ++stats.eh_21_100_edges;
                    if(source_below_mex) ++stats.eh_21_100_from_below_mex;
                    if(source_at_mex) ++stats.eh_21_100_from_at_mex;
                } else {
                    ++stats.eh_101_200_edges;
                    if(source_below_mex) ++stats.eh_101_200_from_below_mex;
                    if(source_at_mex) ++stats.eh_101_200_from_at_mex;
                }
            }
        }
        return stats;
    }

    static size_t repairDegreePercentile(const RepairDegreeStats& stats, double q) {
        if(stats.node_count == 0 || stats.degree_hist.empty()) return 0;
        const size_t target = static_cast<size_t>(std::ceil(q * stats.node_count));
        size_t cumulative = 0;
        for(size_t d = 0; d < stats.degree_hist.size(); ++d) {
            cumulative += stats.degree_hist[d];
            if(cumulative >= std::max<size_t>(target, 1)) return d;
        }
        return stats.degree_hist.size() - 1;
    }

    void writeRepairDegreeDistribution(std::ostream& out) const {
        const auto stats = getRepairDegreeStats();
        out << "repair_degree,node_count,ratio_all_nodes\n";
        const double denom = stats.node_count ? static_cast<double>(stats.node_count) : 1.0;
        for(size_t d = 0; d < stats.degree_hist.size(); ++d) {
            if(stats.degree_hist[d] == 0) continue;
            out << d << "," << stats.degree_hist[d] << ","
                << (static_cast<double>(stats.degree_hist[d]) / denom) << "\n";
        }
    }

    void writeRepairDegreeSummary(std::ostream& out) const {
        const auto stats = getRepairDegreeStats();
        const double nodes = stats.node_count ? static_cast<double>(stats.node_count) : 1.0;
        const double repair_edges = stats.total_repair_edges
            ? static_cast<double>(stats.total_repair_edges) : 1.0;
        out << "metric,value\n";
        out << "MEX," << MEX << "\n";
        out << "node_count," << stats.node_count << "\n";
        out << "nodes_with_repair," << stats.nodes_with_repair << "\n";
        out << "nodes_without_repair," << stats.nodes_without_repair << "\n";
        out << "nodes_below_MEX," << stats.nodes_below_mex << "\n";
        out << "nodes_at_MEX," << stats.nodes_at_mex << "\n";
        out << "nodes_above_MEX," << stats.nodes_above_mex << "\n";
        out << "ratio_nodes_at_MEX," << (static_cast<double>(stats.nodes_at_mex) / nodes) << "\n";
        out << "ratio_nodes_with_repair_at_MEX,"
            << (stats.nodes_with_repair ? static_cast<double>(stats.nodes_at_mex) / stats.nodes_with_repair : 0.0)
            << "\n";
        out << "total_repair_edges," << stats.total_repair_edges << "\n";
        out << "avg_repair_degree_all_nodes," << (static_cast<double>(stats.total_repair_edges) / nodes) << "\n";
        out << "max_repair_degree," << stats.max_repair_degree << "\n";
        out << "p50_repair_degree," << repairDegreePercentile(stats, 0.50) << "\n";
        out << "p90_repair_degree," << repairDegreePercentile(stats, 0.90) << "\n";
        out << "p95_repair_degree," << repairDegreePercentile(stats, 0.95) << "\n";
        out << "p99_repair_degree," << repairDegreePercentile(stats, 0.99) << "\n";
        out << "duplicate_repair_entries," << stats.duplicate_repair_entries << "\n";
        out << "duplicate_repair_entry_ratio,"
            << (static_cast<double>(stats.duplicate_repair_entries) / repair_edges) << "\n";
    }

    void writeRepairEhBySaturation(std::ostream& out) const {
        const auto stats = getRepairDegreeStats();
        out << "eh_group,total_edges,from_source_degree_lt_MEX,from_source_degree_eq_MEX\n";
        out << "EH_0_20," << stats.eh_0_20_edges << ","
            << stats.eh_0_20_from_below_mex << "," << stats.eh_0_20_from_at_mex << "\n";
        out << "EH_21_100," << stats.eh_21_100_edges << ","
            << stats.eh_21_100_from_below_mex << "," << stats.eh_21_100_from_at_mex << "\n";
        out << "EH_101_200," << stats.eh_101_200_edges << ","
            << stats.eh_101_200_from_below_mex << "," << stats.eh_101_200_from_at_mex << "\n";
        out << "EH_INF," << stats.eh_inf_edges << ","
            << stats.eh_inf_from_below_mex << "," << stats.eh_inf_from_at_mex << "\n";
        out << "RFIX," << stats.rfix_edges << ","
            << stats.rfix_from_below_mex << "," << stats.rfix_from_at_mex << "\n";
    }


    std::vector<std::pair<float, id_t> > 
    getNeighborsByHeuristic(std::vector<std::pair<float, id_t> >& neighbor_candidates, const size_t M) {
        if (neighbor_candidates.size() < M) {
            return neighbor_candidates;
        }

        std::vector<std::pair<float, id_t> > return_list;

        for (auto [dist_to_query, cur_id] : neighbor_candidates) {
            if (return_list.size() >= M) {break;}
            bool good = true;

            for (auto [_, exist_id] : return_list) {
                float curdist = getDist(exist_id, cur_id);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back({dist_to_query, cur_id});
            }
        }

        return return_list;
    }


    void HNSWBottomLayerInsertion(T* data, id_t cur_id, size_t efC) {
        size_t NDC = 0;
        auto res = searchKnnBaseGraphConstruction(data, efC, efC, NDC);
        auto neighbors = getNeighborsByHeuristic(res, M);
        
        { // add edge (cur_id, neighbor_id)
            std::unique_lock <std::shared_mutex> lock(node_locks[cur_id]);
            Graph[cur_id].replace_base_graph_neighbors(neighbors);
        }

        // add edge (neighbor_id, cur_id)
        for(auto [_, neighbor_id] : neighbors) {
            std::unique_lock <std::shared_mutex> lock(node_locks[neighbor_id]);
            auto [ids, sz, st] = getBaseGraphNeighbors(neighbor_id);
            if(sz < M0) {
                Graph[neighbor_id].add_base_graph_neighbors(cur_id);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = getDist(neighbor_id, cur_id);
                // Heuristic:
                std::vector<std::pair<float, id_t> > candidates;
                candidates.push_back({d_max, cur_id});
                for (int j = st; j < st + sz; j++) {
                    candidates.push_back({getDist(ids[j], neighbor_id) , ids[j]});
                }
                std::sort(candidates.begin(), candidates.end());
                auto neighbors = getNeighborsByHeuristic(candidates, M0);

                Graph[neighbor_id].replace_base_graph_neighbors(neighbors);
            }
        }
    }

    // The insertion method is HNSW's bottom layer
    void InsertPoint(id_t id, size_t efC, T* vec) {
        if(id >= max_elements) {
            throw std::runtime_error("Error: id > max_elements.");
        }
        SetData(id, vec);
        auto data = getData(id);

        if(n != 0) {
            HNSWBottomLayerInsertion(data, id, efC);
        }
        ++n;
        if(n % 100000 == 0) {
            SetEntryPoint();
        }
    }

    // remove r% NGFix edges
    void PartialRemoveEdges(float r) {
        for(int i = 0; i < n; ++i) {
            std::unique_lock <std::shared_mutex> lock(node_locks[i]);
            auto neighbors = Graph[i].neighbors;
            std::vector<id_t> new_neighbors;
            uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)neighbors);
            uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)neighbors);
            if(ngfix_sz == 0) {
                continue;
            } else {
                int st = ngfix_capacity - ngfix_sz;
                for(int j = st; j < std::max(1, (int32_t)(ngfix_sz*(1.0 - r))) + st; ++j) {
                    new_neighbors.push_back(neighbors[j + 1]);
                }
            }
            Graph[i].replace_ngfix_neighbors(new_neighbors);
        }
    }

    void set_deleted(id_t id) {
        (vecdata + id*size_per_element)[0] = true;
    }
    bool is_deleted(id_t id) {
        return (vecdata + id*size_per_element)[0];
    }

    // You can replace the data corresponding to the deleted ID with a new vector and re-insert it after calling DeleteAllFlagNodesByNGFix.
    void DeleteAllFlagPointsByNGFix(size_t efC_delete = 500, size_t Threads = 32) {
        std::unordered_set<id_t> ids;
        {
            std::unique_lock <std::shared_mutex> lock(delete_lock);
            ids = delete_ids;
            delete_ids.clear();
        }

        // delete correspoding edges
        #pragma omp parallel for schedule(dynamic) num_threads(Threads)
        for(int i = 0; i < n; ++i) {
            if(ids.find(i) != ids.end()) {
                Graph[i].delete_node();
            } else {
                // replace ngfix neighbors and base graph neighbors
                auto [outs, sz, st] = getNeighbors(i);
                uint8_t ngfix_sz = GET_NGFIX_SZ((uint8_t*)outs);
                uint8_t ngfix_capacity = GET_NGFIX_CAPACITY((uint8_t*)outs);
                uint8_t base_sz = sz - ngfix_sz;

                std::vector<id_t> new_neighbors;
                for(int j = ngfix_capacity - ngfix_sz; j < ngfix_capacity; ++j) {
                    if(ids.find(outs[j + 1]) == ids.end()) { // not deleted
                        new_neighbors.push_back(outs[j + 1]);
                    }
                }

                std::vector<std::pair<float, id_t> > new_neighbors2;
                for(int j = ngfix_capacity; j < ngfix_capacity + base_sz; ++j) {
                    if(ids.find(outs[j + 1]) == ids.end()) { // not deleted
                        new_neighbors2.push_back({0, outs[j + 1]});
                    }
                }

                Graph[i].replace_ngfix_neighbors(new_neighbors); 
                Graph[i].replace_base_graph_neighbors(new_neighbors2);
            }
        }

        std::vector<id_t> v_ids;
        for(auto u : ids) {
            v_ids.push_back(u);
        }

        #pragma omp parallel for schedule(dynamic) num_threads(Threads)
        for(int i = 0; i < v_ids.size(); ++i) {
            if(i % 100000 == 0) {
                std::cout<<"delete number "<<i<<"\n";
            }
            int* gt = new int[500];
            AKNNGroundTruth(getData(v_ids[i]), gt, 500, efC_delete);
            NGFix(getData(v_ids[i]), gt, 100, 100);
            delete []gt; 
        }
    }

    // This function is not thread-safe.
    // We only guarantee it can be called concurrently with DeleteAllFlagPointsByNGFix.
    void DeletePointByFlag(id_t id) {
        set_deleted(id);
        {
            std::shared_lock <std::shared_mutex> lock(delete_lock);
            if(id != entry_point) {
                delete_ids.insert(id);
                n = n - 1;
            }
        }
    }
    
    std::unordered_map<id_t, std::vector<id_t> > ComputeGq(int* gt, size_t S) 
    {
        std::unordered_map<id_t, std::vector<id_t> > G;
        std::unordered_set<id_t> Vq;
        for(int i = 0; i < S; ++i){
            Vq.insert(gt[i]);
        }
        for(int i = 0; i < S; ++i){
            int u = gt[i];
            auto [ids, sz, st] = getNeighbors(u);
            for (int j = st; j < st + sz; ++j){
                id_t v = ids[j];
                if(Vq.find(v) == Vq.end()){
                    continue;
                }
                G[u].push_back(v);
            }
        }

        return G;
    }

    std::vector<std::vector<uint16_t> > CalculateHardness(int* gt, size_t Nq, size_t Kh, size_t S) 
    {
        auto Gq = ComputeGq(gt, S);
        std::unordered_map<id_t, uint16_t> p2rank;
        for(int i = 0; i < S; ++i){
            p2rank[gt[i]] = i;
        }
        
        std::vector<std::vector<uint16_t> > H;
        std::bitset<MAX_S> f[S];
        H.resize(Nq);
        for(int i = 0; i < Nq; ++i){
            H[i].resize(Nq, EH_INF);
        }
        for(int h = 0; h < S; ++h){
            f[h][h] = 1;
            if(h < Nq){
                H[h][h] = h;
            }
        }

        for(auto [u, neighbors] : Gq){
            int i = p2rank[u];
            for(auto v : neighbors){
                int j = p2rank[v];
                f[i][j] = 1;
                if(i < Nq && j < Nq){
                    H[i][j] = std::max(i,j);
                }
            }
        }

        for(int h = 0; h < S; ++h){
            for(int i = 0; i < S; ++i){
                /*
                    for(int j = 0; j < S; ++j) {
                        f[i][j] = f[i][h] | f[h][j]
                    }
                */
                auto last = f[i];
                if(f[i][h]){
                    f[i] |= f[h];
                }
                last ^= f[i];
                if(i < Nq && last.count() > 0){
                    for(int j = 0; j < Nq; ++j){
                        if(last[j] == 1){
                            H[i][j] = std::max(std::max(i,j), h);
                        }
                    }
                }
            }
        }
        return H;
    }

    auto getDefectsFixingEdges(
        std::bitset<MAX_Nq> f[],
        std::vector<std::vector<uint16_t> >& H,
        T* query, int* gt, size_t Nq, size_t Kh) {
        
        std::unordered_map<id_t, std::vector<std::pair<id_t, uint16_t> > > new_edges;

        std::vector<std::pair<float, std::pair<int,int> > > vs;
        for(int i = 0; i < Nq; ++i){
            for(int j = 0; j < Nq; ++j){
                if(f[i][j] == 1) {continue;}
                int u = gt[i];
                int v = gt[j]; 
                float d = getDist(u, v);
                vs.push_back({d,{i,j}});
            }
        }
        std::sort(vs.begin(), vs.end());

        for(auto [d, e] : vs){
            int s = e.first;
            int t = e.second;
            if(f[s][t] == 1) {continue;}

            int u = gt[s];
            int v = gt[t];

            new_edges[u].push_back({v, H[s][t]});

            f[s][t] = 1;
            for(int i = 0; i < Nq; ++i){
                if(f[i][s]){
                    f[i] |= f[t];
                }
            }
        }
        return new_edges;
    }

    void NGFix(T* query, int* gt, size_t Nq = 100, size_t Kh = 100) {
        if(Nq > MAX_Nq) {
            throw std::runtime_error("Error: Nq >= MAX_Nq.");
        }
        auto H = CalculateHardness(gt, Nq, Kh, std::min(MAX_S, 2*Nq));
        std::bitset<MAX_Nq> f[Nq];
        for(int i = 0; i < Nq; ++i){
            for(int j = 0; j < Nq; ++j){
                f[i][j] = (H[i][j] <= Kh) ? 1 : 0;
            }
        }
        auto new_edges = getDefectsFixingEdges(f, H, query, gt, Nq, Kh);
        for(auto [u, vs] : new_edges) {
            std::unique_lock <std::shared_mutex> lock(node_locks[u]);
            for(auto [v, eh] : vs) {
                // printf("add edges %d %d\n", u,v);
                Graph[u].add_ngfix_neighbors(v, eh, MEX);
            }
        }
    }

    // return S = {v | delta(v, q) < delta(u, q)}
    std::vector<id_t> searchCloserPoints(T* query_data, size_t ef, size_t u, size_t& ndc) {
        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;
        
        std::vector<id_t> res;
        float dist_u_q = getQueryDist(u, query_data);
        q->push(u, dist_u_q, is_deleted(u));
        q->set_visited(u);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }

            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getQueryDist(candidate_id, query_data);
                    if(dist < dist_u_q && !is_deleted(candidate_id)) {
                        res.push_back(candidate_id);
                    }
                    q->push(candidate_id, dist, is_deleted(candidate_id));
                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        return res;
    }

    /* Experiments show that for most historical queries, 
       a single round of RF is sufficient to bring the search to the vicinity of the query. 
       For simplicity, we perform only one round of RFix. 
    */
    void RFix(T* query, int* gt, size_t Nq = 10, size_t efC = 1500) {
        size_t ndc = 0;
        size_t k = 1;
        auto result = searchKnn(query, k, Nq, ndc);
        
        id_t ANN = result[0].second;
        float d1 = result[0].first;
        float d2 = getQueryDist(gt[Nq - 1], query);
        if(d1 > d2) { // can not reach NG_{Nq,q}
            auto res = searchCloserPoints(query, efC, ANN, ndc);
            
            id_t u = ANN;
            std::vector<std::pair<float, id_t> > candidates;
            for(auto i : res){
                candidates.push_back({getDist(u, i), i});
            }
            std::sort(candidates.begin(), candidates.end());

            auto neighbors = getNeighborsByHeuristic(candidates, 6);

            {
                std::unique_lock <std::shared_mutex> lock(node_locks[u]);
                for(auto [d, v] : neighbors) {
                    Graph[u].add_ngfix_neighbors(v, MAX_S + 1, MEX);
                }
            }
        }
    }

    void AKNNGroundTruth(T* query, int* gt, size_t k, size_t efC) {
        size_t ndc = 0;
        auto result = searchKnn(query, k, efC, ndc);
        for(int i = 0; i < k; ++i) {
            gt[i] = result[i].second;
        }
    }

    // set ep to centroid
    // TODO(): now ep is always 0 when using rabitq
    void SetEntryPoint() {
        T* centroid = new T[dim];
        memset(centroid, 0, sizeof(T)*dim);
        for(int i = 0; i < n; ++i) {
            if(is_deleted(i)) {continue;}
            auto data = getData(i);
            for(int d = 0; d < dim; ++d) {
                centroid[d] += data[d];
            }
        }

        for(int d = 0; d < dim; ++d) {
            centroid[d] /= n;
        }

        id_t ep = 0;
        float min_dis = std::numeric_limits<float>::max();
        for(int i = 0; i < n; ++i) {
            auto dis = space->dist_func(centroid, getData(i));
            if(dis < min_dis) {
                min_dis = dis;
                ep = i;
            }
        }
        entry_point = ep;
        delete []centroid;
    }

    // only used for construction (query_data.bits == base_data.bits)
    std::vector<std::pair<float, id_t> > searchKnnBaseGraphConstruction(T* query_data, size_t k, size_t ef, size_t& ndc) {
        if(ef < k) {
            throw std::runtime_error("Error: efs < k.");
        }
        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        float dist = getDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getBaseGraphNeighbors(current_node_id);

            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }

                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);

        return res;
    }

    std::vector<std::pair<float, id_t> > searchKnn(T* query_data, size_t k, size_t ef, size_t& ndc) {
        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        // Search_PriorityQueue q0(ef, visited_list_pool_);
        // Search_Array q0(ef, visited_list_pool_);
        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;
        
        float dist = getQueryDist(entry_point, query_data);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        while (!q->is_empty()) {
            if (hop > hop_limit) {
                break;
            }
            std::pair<float, id_t> current_node_pair = q->get_next_id();
            id_t current_node_id = current_node_pair.second;

            float candidate_dist = -current_node_pair.first;
            bool flag_stop_search;
            flag_stop_search = candidate_dist > q->get_dist_bound();

            if (flag_stop_search) {
                break;
            }
            hop += 1;
            // std::cout<<current_node_id<<" "<<candidate_dist<<"\n";
            std::shared_lock <std::shared_mutex> lock(node_locks[current_node_id]);
            auto [outs, sz, st] = getNeighbors(current_node_id);
            for (int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i+1]);
                }
                if (!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float dist = getQueryDist(candidate_id, query_data);
                    q->push(candidate_id, dist, is_deleted(candidate_id));

                    ndc += 1;
                }
            }
        }
        q->releaseVisitedList();
        auto res = q->get_result(k);
        return res;
    }


    // ============================================================
    // RoarGraph-style history table loader
    // Format:
    // node_id frequency
    // 0 0.12
    // 1 0.01
    // ============================================================
    void LoadHistoryTable(const std::string& path)
    {
        history_frequency.assign(max_elements, -1.0f);
        std::ifstream in(path);
        if(!in.is_open()) {
            throw std::runtime_error("Cannot open history table: " + path);
        }

        id_t id;
        float freq;
        while(in >> id >> freq) {
            if(id < history_frequency.size()) {
                history_frequency[id] = freq;
            }
        }
        in.close();
    }


    // ============================================================
    // Baseline NGFix search with observation-only statistics.
    // searchKnn() itself remains untouched; this function mirrors its
    // search decisions exactly and only adds counters/edge classification.
    // ============================================================
    std::vector<std::pair<float, id_t> >
    searchKnnWithStats(T* query_data, size_t k, size_t ef, size_t& ndc,
                       TwoHopSearchStats& stats)
    {
        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        struct DiscoveryOrigin {
            id_t source;
            int neighbor_index;
        };
        // Only admitted nodes need an origin. This is instrumentation-only;
        // latency from this profiling path should not be used for final system claims.
        std::unordered_map<id_t, DiscoveryOrigin> admitted_origin;
        admitted_origin.reserve(ef * 4 + 16);

        float dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        if(stats.collect_node_ids) stats.scored_nodes.insert(entry_point);
        q->push(entry_point, dist, is_deleted(entry_point));
        q->set_visited(entry_point);

        while(!q->is_empty()) {
            if(hop > hop_limit) {
                ++stats.hop_limit_stop_count;
                break;
            }

            auto current_node_pair = q->get_next_id();
            id_t cur = current_node_pair.second;
            float candidate_dist = -current_node_pair.first;
            if(candidate_dist > q->get_dist_bound()) break;

            // cur is actually expanded now (not merely admitted). Attribute this
            // event to the unique discovery edge; entry point has no such edge.
            auto oit = admitted_origin.find(cur);
            if(oit != admitted_origin.end()) {
                recordExpandedFromEdge(stats.first_hop_edges,
                                       oit->second.source,
                                       oit->second.neighbor_index);
                admitted_origin.erase(oit);
            }

            ++hop;
            ++stats.cur_pop_count;
            if(stats.collect_node_ids) stats.expanded_nodes.insert(cur);

            std::shared_lock<std::shared_mutex> lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);
            for(int i = st; i < st + sz; ++i) {
                id_t candidate_id = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i + 1]);
                }

                if(!q->is_visited(candidate_id)) {
                    q->set_visited(candidate_id);
                    float d = getQueryDist(candidate_id, query_data);
                    ++ndc;
                    ++stats.distance_cmps;
                    ++stats.mid_distance_cmps;
                    if(stats.collect_node_ids) stats.scored_nodes.insert(candidate_id);
                    recordScoredEdge(stats.first_hop_edges, cur, i);

                    // Mirror Search_QuadHeap::push()'s admission predicate exactly.
                    const bool admitted =
                        q->top_candidates.size() < q->L || q->get_dist_bound() > d;

                    q->push(candidate_id, d, is_deleted(candidate_id));
                    if(admitted) {
                        recordAdmittedEdge(stats.first_hop_edges, cur, i);
                        admitted_origin[candidate_id] = {cur, i};
                        ++stats.mid_enqueue_count;
                    }
                }
            }
        }

        q->releaseVisitedList();
        return q->get_result(k);
    }


    // ============================================================
    // Distance Adaptive Beam Search (Al-Jazzazi et al., NeurIPS 2025,
    // Algorithm 2). Unlike fixed-ef beam search, DABS keeps exactly the
    // k best discovered results and admits/expands candidates inside the
    // multiplicative distance window (1 + gamma) * d_k.
    //
    // The stored distance is transformed before applying that window:
    //   NORMALIZED_IP: sqrt(2 * max(0, 1 - q dot x))
    //   SQUARED_L2:    sqrt(max(0, squared_l2))
    // Both produce the actual Euclidean metric required by DABS. For IP,
    // callers must first verify that both base and query vectors have unit
    // norm; raw 1-dot is neither cosine distance nor a metric otherwise.
    // ============================================================
    enum class DABSDistanceTransform : uint8_t {
        NORMALIZED_IP = 0,
        SQUARED_L2 = 1
    };

    static float dabsSquaredMetricDistance(float stored_distance,
                                           DABSDistanceTransform transform)
    {
        const float nonnegative = std::max(0.0f, stored_distance);
        if(transform == DABSDistanceTransform::NORMALIZED_IP)
            return 2.0f * nonnegative;
        return nonnegative;
    }

    std::vector<std::pair<float, id_t> >
    searchKnnDABSWithStats(T* query_data,
                           size_t k,
                           float gamma,
                           DABSDistanceTransform transform,
                           size_t& ndc,
                           TwoHopSearchStats& stats)
    {
        if(k == 0) throw std::runtime_error("DABS requires k > 0.");
        if(gamma < 0.0f)
            throw std::runtime_error("DABS requires gamma >= 0.");

        using HeapItem = std::pair<float, id_t>;
        // candidates: closest distance first; best: farthest of the current
        // k best discovered, matching Algorithm 2 in the DABS paper.
        std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> candidates;
        std::priority_queue<HeapItem> best;

        VisitedList* vl = visited_list_pool_->getFreeVisitedList();
        vl_type* visited = vl->mass;
        const vl_type visited_tag = vl->curV;

        struct DiscoveryOrigin {
            id_t source;
            int neighbor_index;
        };
        std::unordered_map<id_t, DiscoveryOrigin> admitted_origin;
        admitted_origin.reserve(k * 4 + 16);

        const float squared_factor = (1.0f + gamma) * (1.0f + gamma);
        const auto squared_metric_distance = [transform](float d) {
            return dabsSquaredMetricDistance(d, transform);
        };

        const float entry_dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        if(stats.collect_node_ids) stats.scored_nodes.insert(entry_point);
        visited[entry_point] = visited_tag;
        candidates.emplace(entry_dist, entry_point);
        if(!is_deleted(entry_point)) best.emplace(entry_dist, entry_point);

        while(!candidates.empty()) {
            const auto [candidate_dist, cur] = candidates.top();
            candidates.pop();

            if(best.size() == k &&
               squared_factor * squared_metric_distance(best.top().first) <=
                   squared_metric_distance(candidate_dist)) {
                break;
            }

            auto oit = admitted_origin.find(cur);
            if(oit != admitted_origin.end()) {
                recordExpandedFromEdge(stats.first_hop_edges,
                                       oit->second.source,
                                       oit->second.neighbor_index);
                admitted_origin.erase(oit);
            }

            ++stats.cur_pop_count;
            if(stats.collect_node_ids) stats.expanded_nodes.insert(cur);

            std::shared_lock<std::shared_mutex> lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);
            for(int i = st; i < st + sz; ++i) {
                const id_t candidate_id = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1)
                    __builtin_prefetch(visited + outs[i + 1], 0, 1);

                if(visited[candidate_id] == visited_tag) continue;
                visited[candidate_id] = visited_tag;

                const float d = getQueryDist(candidate_id, query_data);
                ++ndc;
                ++stats.distance_cmps;
                ++stats.mid_distance_cmps;
                if(stats.collect_node_ids) stats.scored_nodes.insert(candidate_id);
                recordScoredEdge(stats.first_hop_edges, cur, i);

                // Follow Algorithm 2 exactly: use the current d_k for the
                // admission test, then insert into both heaps and trim B.
                const bool admitted =
                    best.size() < k ||
                    squared_metric_distance(d) <
                        squared_factor * squared_metric_distance(best.top().first);
                if(!admitted) continue;

                candidates.emplace(d, candidate_id);
                if(!is_deleted(candidate_id)) {
                    best.emplace(d, candidate_id);
                    if(best.size() > k) best.pop();
                }

                recordAdmittedEdge(stats.first_hop_edges, cur, i);
                admitted_origin[candidate_id] = {cur, i};
                ++stats.mid_enqueue_count;
            }
        }

        visited_list_pool_->releaseVisitedList(vl);

        std::vector<std::pair<float, id_t>> result;
        result.reserve(std::min(k, best.size()));
        while(!best.empty()) {
            result.push_back(best.top());
            best.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }


    // ============================================================
    // NGFix + RoarGraph-style jump search.
    // For every popped cur:
    //   1) score one-hop mid (with cache);
    //   2) if mid <= cur, enqueue mid and do not scan N(mid);
    //   3) if mid > cur, scan mid -> nn;
    //      nn < cur is a valid jump candidate;
    //   4) if any valid jump exists, enqueue valid nn and skip mid;
    //      otherwise fall back to enqueueing mid.
    // ============================================================
    std::vector<std::pair<float, id_t> >
    searchKnnTwoHop(T* query_data, size_t k, size_t ef, size_t& ndc,
                    TwoHopSearchStats* stats_ptr = nullptr)
    {
        TwoHopSearchStats local_stats;
        TwoHopSearchStats& stats = stats_ptr ? *stats_ptr : local_stats;

        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        enum NodeState : uint8_t { UNSEEN = 0, SCORED = 1, ENQUEUED = 2 };
        std::vector<uint8_t> state(max_elements, UNSEEN);
        std::vector<float> dist_cache(max_elements, 0.0f);

        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        const float ep_dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        dist_cache[entry_point] = ep_dist;
        state[entry_point] = ENQUEUED;
        q->push(entry_point, ep_dist, is_deleted(entry_point));
        q->set_visited(entry_point);

        auto score_node = [&](id_t x, bool is_mid, bool* newly_computed = nullptr) -> float {
            bool did_compute = false;
            if(state[x] == UNSEEN) {
                dist_cache[x] = getQueryDist(x, query_data);
                state[x] = SCORED;
                ++ndc;
                ++stats.distance_cmps;
                if(is_mid) ++stats.mid_distance_cmps;
                else ++stats.nn_distance_cmps;
                did_compute = true;
            } else {
                ++stats.cache_hits;
            }
            if(newly_computed) *newly_computed = did_compute;
            return dist_cache[x];
        };

        auto enqueue_node = [&](id_t x, float d) -> bool {
            if(state[x] == ENQUEUED) return false;
            state[x] = ENQUEUED;
            q->set_visited(x);
            q->push(x, d, is_deleted(x));
            return true;
        };

        while(!q->is_empty()) {
            if(hop > hop_limit) {
                ++stats.hop_limit_stop_count;
                break;
            }

            auto current_node_pair = q->get_next_id();
            const id_t cur = current_node_pair.second;
            const float cur_dist = -current_node_pair.first;
            if(cur_dist > q->get_dist_bound()) break;

            ++hop;
            ++stats.cur_pop_count;

            std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);

            for(int i = st; i < st + sz; ++i) {
                const id_t mid = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i + 1]);
                    __builtin_prefetch(getData(outs[i + 1]), 0, 1);
                }

                if(state[mid] == ENQUEUED) {
                    ++stats.cache_hits;
                    ++stats.skipped_mid_count;
                    continue;
                }

                const float mid_dist = score_node(mid, true);

                if(mid_dist <= cur_dist) {
                    ++stats.direct_mid_count;
                    if(enqueue_node(mid, mid_dist)) {
                        ++stats.mid_enqueue_count;
                        recordEnqueuedEdge(stats.first_hop_edges, cur, i);
                    }
                    continue;
                }

                ++stats.twohop_mid_scan_count;
                bool has_jump = false;
                struct JumpCandidate {
                    id_t id;
                    float dist;
                    int neighbor_index;
                };
                std::vector<JumpCandidate> jump_candidates;
                size_t fresh_nn_distance_cmps_for_mid = 0;

                std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
                auto [nn_outs, nn_sz, nn_st] = getNeighbors(mid);
                jump_candidates.reserve(nn_sz);

                for(int j = nn_st; j < nn_st + nn_sz; ++j) {
                    const id_t nn = nn_outs[j];
                    ++stats.checked_nn_count;
                    recordExaminedEdge(stats.second_hop_edges, mid, j);
                    const SearchEdgeType first_edge_type = getSearchEdgeType(cur, i);
                    const SearchEdgeType second_edge_type = getSearchEdgeType(mid, j);
                    recordEdgePair(stats.twohop_examined_pairs, first_edge_type, second_edge_type);

                    if(j < nn_st + nn_sz - 1) {
                        q->prefetch_visited_list(nn_outs[j + 1]);
                        __builtin_prefetch(getData(nn_outs[j + 1]), 0, 1);
                    }

                    if(state[nn] == ENQUEUED) {
                        ++stats.cache_hits;
                        continue;
                    }

                    bool nn_computed = false;
                    const float nn_dist = score_node(nn, false, &nn_computed);
                    if(nn_computed) {
                        ++fresh_nn_distance_cmps_for_mid;
                    }
                    if(nn_dist < cur_dist) {
                        has_jump = true;
                        ++stats.jump_candidate_count;
                        recordEdgePair(stats.jump_pairs, first_edge_type, second_edge_type);
                        jump_candidates.push_back({nn, nn_dist, j});
                    }
                }

                if(has_jump) {
                    ++stats.jump_mid_count;
                    stats.nn_jump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    std::sort(jump_candidates.begin(), jump_candidates.end(),
                              [](const JumpCandidate& a, const JumpCandidate& b) {
                                  return a.dist < b.dist;
                              });
                    for(const auto& jc : jump_candidates) {
                        if(enqueue_node(jc.id, jc.dist)) {
                            ++stats.nn_enqueue_count;
                            recordEnqueuedEdge(stats.second_hop_edges, mid, jc.neighbor_index);
                        }
                    }
                    // mid intentionally skipped
                } else {
                    ++stats.failed_jump_mid_count;
                    stats.nn_nojump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    if(enqueue_node(mid, mid_dist)) {
                        ++stats.mid_enqueue_count;
                        recordEnqueuedEdge(stats.first_hop_edges, cur, i);
                    }
                }
            }
        }

        q->releaseVisitedList();
        return q->get_result(k);
    }


    // ============================================================
    // Bound-triggered two-hop bridge search.
    //
    // This mode is intentionally much closer to the normal beam search than
    // searchKnnTwoHop():
    //   1) score cur -> mid exactly as baseline;
    //   2) if mid satisfies the CURRENT beam admission predicate, enqueue it
    //      normally and DO NOT inspect N(mid);
    //   3) only if mid is rejected by the current beam bound, inspect mid -> nn;
    //   4) a second-hop nn is a jump candidate only if nn itself satisfies the
    //      CURRENT beam admission predicate; admitted nn nodes are inserted into
    //      the normal beam, while the rejected mid remains skipped.
    //
    // Thus the only extra exploration relative to baseline is behind a node
    // that baseline would have rejected at that exact search state. This makes
    // the experiment a much cleaner test of "bridge nodes hidden behind the
    // beam bound" than the old mid_dist > cur_dist trigger.
    // ============================================================
    std::vector<std::pair<float, id_t> >
    searchKnnBoundTwoHop(T* query_data, size_t k, size_t ef, size_t& ndc,
                         TwoHopSearchStats* stats_ptr = nullptr)
    {
        TwoHopSearchStats local_stats;
        TwoHopSearchStats& stats = stats_ptr ? *stats_ptr : local_stats;
        const auto& gate = bound_twohop_gate;
        gate.validate();
        jump_policy.validate();
        ObservedDistanceTopK observed_topk(gate.cur_topk);
        const bool deferred = gate.schedule == JumpSchedule::AFTER_ONEHOP;
        ObservedDistanceTopK state_topk(deferred ? k : 0);
        JumpProgressWindow progress(gate.stall_window);
        const bool partial_nn = gate.nn_selection != JumpNNSelection::FULL;
        const bool need_cheap = gate.nn_selection == JumpNNSelection::CHEAP || stats.collect_nn_events;
        JumpCheapQuery cheap_query;
        if(need_cheap) {
            if constexpr(std::is_same_v<T, float>) {
                const bool ip = dynamic_cast<IPSpace_float*>(query_space) != nullptr;
                if(!ip && !dynamic_cast<L2Space_float*>(query_space))
                    throw std::invalid_argument("NN cheap selection/trace requires raw float IP or L2.");
                cheap_query.prepare(query_data, dim, gate.nn_cheap_dims, ip);
                stats.gate.nn_query_dims_inspected += dim;
            } else throw std::invalid_argument("NN cheap selection/trace requires float vectors.");
        }

        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        struct DiscoveryOrigin {
            id_t source;
            int neighbor_index;
            bool second_hop;
        };
        std::unordered_map<id_t, DiscoveryOrigin> admitted_origin;
        admitted_origin.reserve(ef * 4 + 16);

        // Only populated for event tracing. These sets let us attribute a
        // later first-hop visited skip to the second-hop probe that first
        // scored that node, without changing normal search behavior.
        std::unordered_set<id_t> scored_by_success_probe;
        std::unordered_set<id_t> scored_by_failed_probe;
        std::unordered_set<id_t> attributed_success_probe_skips;
        std::unordered_set<id_t> attributed_failed_probe_skips;
        std::unordered_set<id_t> attributed_admitted_jump_skips;

        ObservedDistanceTopK trace_topk(stats.collect_search_state ? k : 0);
        size_t trace_result_updates=0,trace_topk_updates=0,trace_last_topk_step=0;
        std::unordered_map<id_t,size_t> trace_depth;
        auto trace_admission = [&](id_t id,float distance,size_t depth) {
            if(!stats.collect_search_state) return;
            trace_depth[id]=depth;
            if(is_deleted(id)) return;
            ++trace_result_updates;
            if(trace_topk.observe(distance,false)) {
                ++trace_topk_updates;trace_last_topk_step=hop;
            }
        };

        const float ep_dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        if(stats.collect_node_ids) stats.scored_nodes.insert(entry_point);
        if(gate.cur_topk) observed_topk.observe(ep_dist, is_deleted(entry_point));
        state_topk.observe(ep_dist, is_deleted(entry_point));
        q->push(entry_point, ep_dist, is_deleted(entry_point));
        q->set_visited(entry_point);
        trace_admission(entry_point,ep_dist,0);

        while(!q->is_empty()) {
            if(hop > hop_limit) {
                ++stats.hop_limit_stop_count;
                break;
            }

            auto current_node_pair = q->get_next_id();
            const id_t cur = current_node_pair.second;
            const float cur_dist = -current_node_pair.first;
            if(cur_dist > q->get_dist_bound()) break;

            auto oit = admitted_origin.find(cur);
            if(oit != admitted_origin.end()) {
                if(oit->second.second_hop) {
                    recordExpandedFromEdge(stats.second_hop_edges,
                                           oit->second.source,
                                           oit->second.neighbor_index);
                } else {
                    recordExpandedFromEdge(stats.first_hop_edges,
                                           oit->second.source,
                                           oit->second.neighbor_index);
                }
                admitted_origin.erase(oit);
            }

            ++hop;
            ++stats.cur_pop_count;
            if(stats.collect_node_ids) stats.expanded_nodes.insert(cur);

            std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);
            size_t probed_mids_for_cur = 0;
            size_t nn_for_cur = 0, onehop_changes = 0;
            std::vector<PendingJumpMid> pending;
            if(deferred) pending.reserve(sz);
            auto allows_mid = [&](id_t mid, float mid_dist, int i) {
                const SearchEdgeType first_edge_type = getSearchEdgeType(cur, i);
                const uint16_t first_edge_eh = getSearchEdgeEh(cur, i);
                const float bound_at_mid_reject = q->get_dist_bound();
                const double gate_scale = BoundTwoHopGate::scale(bound_at_mid_reject);
                if(hop < gate.min_hops) {
                    ++stats.gate.blocked_hop;
                    return false;
                }
                if((gate.cur_topk && is_deleted(cur)) || !observed_topk.allows(cur_dist) ||
                   (gate.cur_margin > 0 &&
                    bound_at_mid_reject - static_cast<double>(cur_dist) < gate.cur_margin * gate_scale)) {
                    ++stats.gate.blocked_cur;
                    return false;
                }
                if(gate.has_mid_gap() &&
                   mid_dist - static_cast<double>(bound_at_mid_reject) > gate.mid_gap * gate_scale) {
                    ++stats.gate.blocked_gap;
                    return false;
                }
                if(!gate.allows_detour(cur_dist, mid_dist, bound_at_mid_reject)) {
                    ++stats.gate.blocked_detour;
                    return false;
                }
                if(!jumpEdgeAllowed(gate.first_edge, static_cast<unsigned>(first_edge_type), first_edge_eh)) {
                    ++stats.gate.blocked_edge;
                    return false;
                }
                if((gate.max_mids_per_cur && probed_mids_for_cur >= gate.max_mids_per_cur) ||
                   (gate.max_nn_per_query && stats.nn_distance_cmps >= gate.max_nn_per_query)) {
                    ++stats.gate.blocked_budget;
                    return false;
                }
                if(jump_policy.active() && !jump_policy.allows(cur,
                       cur_dist, mid_dist, bound_at_mid_reject,
                       hop, ef, static_cast<unsigned>(first_edge_type), first_edge_eh,
                       stats.gate)) {
                    ++stats.gate.blocked_policy;
                    return false;
                }
                if(gate.max_nn_per_cur && nn_for_cur >= gate.max_nn_per_cur) {
                    ++stats.gate.blocked_batch_budget;
                    return false;
                }
                return true;
            };
            auto probe_mid = [&](id_t mid, float mid_dist, int i) {
                const SearchEdgeType first_edge_type = getSearchEdgeType(cur, i);
                const uint16_t first_edge_eh = getSearchEdgeEh(cur, i);
                const float bound_at_mid_reject = q->get_dist_bound();
                ++stats.twohop_mid_scan_count;
                const size_t trace_probe_id=stats.twohop_mid_scan_count;
                BoundSearchSnapshot trace_state;
                if(stats.collect_search_state)
                    trace_state=captureBoundSearchSnapshot(*q,k,hop,stats.distance_cmps,
                        stats.mid_distance_cmps,stats.nn_distance_cmps,trace_result_updates,
                        trace_topk_updates,trace_last_topk_step,trace_depth[cur],ep_dist);
                if(first_edge_type == SearchEdgeType::HNSW_BASE) ++stats.gate.probed_base_mids;
                else if(first_edge_type == SearchEdgeType::RFIX) ++stats.gate.probed_rfix_mids;
                else ++stats.gate.probed_ngfix_mids;
                ++probed_mids_for_cur;
                size_t fresh_nn_distance_cmps_for_mid = 0;
                size_t checked_nn_for_mid = 0;
                std::vector<id_t> fresh_nn_ids_for_mid;
                bool has_admitted_jump = false;

                struct JumpCandidate {
                    id_t id;
                    float dist;
                    int neighbor_index;
                    SearchEdgeType second_edge_type;
                    uint16_t second_edge_eh;
                };
                std::vector<JumpCandidate> candidates;

                std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
                auto [nn_outs, nn_sz, nn_st] = getNeighbors(mid);
                candidates.reserve(nn_sz);
                if(stats.collect_jump_events) fresh_nn_ids_for_mid.reserve(nn_sz);

                // Census only for partial selectors or shadow diagnostics. FULL
                // with tracing disabled retains the old loop and score order.
                std::vector<JumpNNOption> nn_pool;
                std::vector<typename TwoHopSearchStats::BoundNNEvent> nn_trace;
                std::vector<int> trace_index(stats.collect_nn_events ? nn_sz : 0, -1);
                std::vector<bool> chosen(partial_nn ? nn_sz : 0, false);
                if(partial_nn || stats.collect_nn_events) {
                    std::unordered_set<id_t> seen;
                    for(int j=nn_st; j<nn_st+nn_sz; ++j) {
                        const id_t nn=nn_outs[j];
                        if(partial_nn) {
                            ++stats.checked_nn_count; ++checked_nn_for_mid;
                            recordExaminedEdge(stats.second_hop_edges, mid, j);
                            recordEdgePair(stats.twohop_examined_pairs, first_edge_type, getSearchEdgeType(mid,j));
                        }
                        if(q->is_visited(nn)) continue;
                        if(!jumpEdgeAllowed(gate.second_edge, static_cast<unsigned>(getSearchEdgeType(mid,j)), getSearchEdgeEh(mid,j))) {
                            if(partial_nn) ++stats.gate.skipped_nn_edge;
                            continue;
                        }
                        if(!seen.insert(nn).second) continue;
                        const float cheap=need_cheap ? cheap_query.score(reinterpret_cast<const float*>(getData(nn))) : 0;
                        if(need_cheap) {
                            if(gate.nn_selection == JumpNNSelection::CHEAP) {
                                ++stats.gate.nn_cheap_scores; stats.gate.nn_cheap_products += gate.nn_cheap_dims;
                            } else {
                                ++stats.gate.nn_shadow_scores; stats.gate.nn_shadow_products += gate.nn_cheap_dims;
                            }
                        }
                        const auto key=jumpNNRandomKey(gate.nn_seed, cur, mid, nn);
                        nn_pool.push_back({nn,j,cheap,key});
                        if(stats.collect_nn_events) {
                            typename TwoHopSearchStats::BoundNNEvent ev;
                            ev.probe_id=stats.twohop_mid_scan_count; ev.hop=hop;
                            ev.cur=cur; ev.mid=mid; ev.nn=nn; ev.adjacency_ordinal=j-nn_st;
                            ev.first_type=static_cast<unsigned>(first_edge_type);
                            ev.second_type=static_cast<unsigned>(getSearchEdgeType(mid,j)); ev.second_eh=getSearchEdgeEh(mid,j);
                            ev.cheap_score=cheap; ev.random_key=key;
                            trace_index[j-nn_st]=static_cast<int>(nn_trace.size()); nn_trace.push_back(ev);
                        }
                    }
                    const auto indices=selectJumpNNIndices(nn_pool,gate.nn_selection,gate.nn_limit);
                    stats.gate.nn_pool_targets += nn_pool.size();
                    stats.gate.nn_selected_targets += indices.size();
                    if(partial_nn) for(int j:indices) chosen[j-nn_st]=true;
                    for(auto& ev:nn_trace) ev.selected=!partial_nn || chosen[ev.adjacency_ordinal];
                }

                for(int j = nn_st; j < nn_st + nn_sz; ++j) {
                    if(partial_nn && !chosen[j-nn_st]) continue;
                    if(gate.max_nn_per_query && stats.nn_distance_cmps >= gate.max_nn_per_query) break;
                    if(gate.max_nn_per_mid && fresh_nn_distance_cmps_for_mid >= gate.max_nn_per_mid) {
                        ++stats.gate.nn_mid_cap_hits; break;
                    }
                    if(gate.max_nn_per_cur && nn_for_cur >= gate.max_nn_per_cur) {
                        ++stats.gate.nn_cur_cap_hits; break;
                    }
                    const id_t nn = nn_outs[j];
                    const SearchEdgeType second_edge_type = getSearchEdgeType(mid, j);
                    if(!partial_nn) {
                        ++stats.checked_nn_count; ++checked_nn_for_mid;
                        recordExaminedEdge(stats.second_hop_edges, mid, j);
                        recordEdgePair(stats.twohop_examined_pairs, first_edge_type, second_edge_type);
                    }

                    if(j < nn_st + nn_sz - 1) {
                        q->prefetch_visited_list(nn_outs[j + 1]);
                        __builtin_prefetch(getData(nn_outs[j + 1]), 0, 1);
                    }

                    if(q->is_visited(nn)) continue;

                    if(!jumpEdgeAllowed(gate.second_edge, static_cast<unsigned>(second_edge_type),
                                        getSearchEdgeEh(mid, j))) {
                        ++stats.gate.skipped_nn_edge;
                        continue;
                    }

                    // A fresh second-hop score: unlike examined_pair, this
                    // denominator excludes nn targets already visited earlier.
                    recordEdgePair(stats.twohop_scored_pairs,
                                   first_edge_type,
                                   second_edge_type);
                    q->set_visited(nn);
                    const float nn_dist = getQueryDist(nn, query_data);
                    ++ndc;
                    ++stats.distance_cmps;
                    ++stats.nn_distance_cmps;
                    ++nn_for_cur;
                    if(state_topk.observe(nn_dist, is_deleted(nn))) ++stats.gate.twohop_topk_changes;
                    ++fresh_nn_distance_cmps_for_mid;
                    if(stats.collect_jump_events) fresh_nn_ids_for_mid.push_back(nn);
                    if(stats.collect_node_ids) stats.scored_nodes.insert(nn);
                    if(gate.cur_topk) observed_topk.observe(nn_dist, is_deleted(nn));
                    recordScoredEdge(stats.second_hop_edges, mid, j);

                    candidates.push_back({nn, nn_dist, j, second_edge_type, getSearchEdgeEh(mid, j)});
                    if(stats.collect_nn_events && trace_index[j-nn_st]>=0) {
                        auto& ev=nn_trace[trace_index[j-nn_st]];
                        ev.scored=true; ev.exact_dist=nn_dist;
                    }
                }

                // Process promising second-hop nodes first. The actual jump
                // condition below is still the live beam admission predicate,
                // so every recorded jump is a node that the beam accepts.
                std::sort(candidates.begin(), candidates.end(),
                          [](const JumpCandidate& a, const JumpCandidate& b) {
                              return a.dist < b.dist;
                          });

                size_t admitted_nn_for_mid = 0;
                id_t best_admitted_nn = std::numeric_limits<id_t>::max();
                float best_admitted_nn_dist = std::numeric_limits<float>::infinity();
                size_t fresh_rank = 0;
                for(const auto& jc : candidates) {
                    ++fresh_rank;
                    const float bound_before_nn_admit = q->get_dist_bound();
                    const bool nn_admitted =
                        q->top_candidates.size() < q->L || q->get_dist_bound() > jc.dist;
                    if(stats.collect_nn_events && trace_index[jc.neighbor_index-nn_st]>=0) {
                        auto& ev=nn_trace[trace_index[jc.neighbor_index-nn_st]];
                        ev.exact_rank=fresh_rank; ev.admitted=nn_admitted;
                    }
                    if(!nn_admitted) continue;

                    q->push(jc.id, jc.dist, is_deleted(jc.id));
                    if(stats.collect_search_state) trace_admission(jc.id,jc.dist,trace_depth[cur]+2);
                    recordAdmittedEdge(stats.second_hop_edges, mid, jc.neighbor_index);
                    admitted_origin[jc.id] = {mid, jc.neighbor_index, true};
                    stats.jump_admitted_nodes.insert(jc.id);
                    auto parent_jump = stats.bridge_root_by_node.find(cur);
                    if(parent_jump != stats.bridge_root_by_node.end() &&
                       parent_jump->second != jc.id) {
                        stats.jump_parent_root[jc.id] = parent_jump->second;
                    }
                    stats.bridge_root_by_node[jc.id] = jc.id;
                    ++stats.nn_enqueue_count;
                    ++stats.jump_candidate_count;
                    has_admitted_jump = true;
                    ++admitted_nn_for_mid;
                    if(jc.dist < best_admitted_nn_dist) {
                        best_admitted_nn = jc.id;
                        best_admitted_nn_dist = jc.dist;
                    }
                    recordEdgePair(stats.jump_pairs,
                                   first_edge_type,
                                   jc.second_edge_type);

                    if(stats.collect_jump_events) {
                        typename TwoHopSearchStats::BoundJumpEvent ev;
                        ev.cur = cur;
                        ev.mid = mid;
                        ev.nn = jc.id;
                        ev.hop = hop;
                        ev.cur_dist = cur_dist;
                        ev.mid_dist = mid_dist;
                        ev.nn_dist = jc.dist;
                        ev.bound_at_mid_reject = bound_at_mid_reject;
                        ev.bound_before_nn_admit = bound_before_nn_admit;
                        ev.first_edge_type = first_edge_type;
                        ev.second_edge_type = jc.second_edge_type;
                        ev.first_edge_eh = first_edge_eh;
                        ev.second_edge_eh = jc.second_edge_eh;
                        ev.rank_in_fresh_nn_by_query_dist = fresh_rank;
                        ev.mid_checked_nn = checked_nn_for_mid;
                        ev.mid_fresh_nn_cmps = fresh_nn_distance_cmps_for_mid;
                        stats.bound_jump_events.push_back(ev);
                    }
                }

                if(stats.collect_nn_events) {
                    const size_t planned=gate.nn_selection == JumpNNSelection::FULL ? nn_pool.size() : std::min(nn_pool.size(),gate.nn_limit);
                    for(auto& ev:nn_trace) {
                        ev.pool_size=nn_pool.size(); ev.scored_size=candidates.size();
                        ev.censored=candidates.size()<planned;
                        stats.bound_nn_events.push_back(ev);
                    }
                }
                if(has_admitted_jump) {
                    ++stats.jump_mid_count;
                    stats.nn_jump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    if(stats.collect_jump_events) {
                        for(const id_t x : fresh_nn_ids_for_mid)
                            scored_by_success_probe.insert(x);
                    }
                } else {
                    ++stats.failed_jump_mid_count;
                    stats.nn_nojump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    if(stats.collect_jump_events) {
                        for(const id_t x : fresh_nn_ids_for_mid)
                            scored_by_failed_probe.insert(x);
                    }
                }

                if(stats.collect_jump_events) {
                    typename TwoHopSearchStats::BoundMidProbeEvent mev;
                    mev.probe_id=trace_probe_id;
                    mev.state=trace_state;
                    mev.cur = cur;
                    mev.mid = mid;
                    mev.hop = hop;
                    mev.cur_dist = cur_dist;
                    mev.mid_dist = mid_dist;
                    mev.bound_at_reject = bound_at_mid_reject;
                    mev.first_edge_type = first_edge_type;
                    mev.first_edge_eh = first_edge_eh;
                    mev.checked_nn = checked_nn_for_mid;
                    mev.fresh_nn_cmps = fresh_nn_distance_cmps_for_mid;
                    mev.admitted_nn = admitted_nn_for_mid;
                    if(!candidates.empty()) {
                        mev.best_scored_nn = candidates.front().id;
                        mev.best_scored_nn_dist = candidates.front().dist;
                        mev.best_scored_second_edge_type = candidates.front().second_edge_type;
                        mev.best_scored_second_edge_eh = candidates.front().second_edge_eh;
                    }
                    mev.best_admitted_nn = best_admitted_nn;
                    mev.best_admitted_nn_dist = best_admitted_nn_dist;
                    stats.bound_mid_probe_events.push_back(mev);
                }

            };

            for(int i = st; i < st + sz; ++i) {
                const id_t mid = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i + 1]);
                    __builtin_prefetch(getData(outs[i + 1]), 0, 1);
                }

                if(q->is_visited(mid)) {
                    ++stats.skipped_mid_count;
                    if(stats.collect_jump_events) {
                        if(scored_by_success_probe.count(mid) &&
                           attributed_success_probe_skips.insert(mid).second)
                            ++stats.skipped_mid_from_success_probe;
                        if(scored_by_failed_probe.count(mid) &&
                           attributed_failed_probe_skips.insert(mid).second)
                            ++stats.skipped_mid_from_failed_probe;
                        if(stats.jump_admitted_nodes.count(mid) &&
                           attributed_admitted_jump_skips.insert(mid).second)
                            ++stats.skipped_mid_from_admitted_jump;
                    }
                    continue;
                }

                // Match baseline semantics: once a node is scored it is marked
                // visited, even if the beam rejects it.
                q->set_visited(mid);
                const float mid_dist = getQueryDist(mid, query_data);
                ++ndc;
                ++stats.distance_cmps;
                ++stats.mid_distance_cmps;
                if(stats.collect_node_ids) stats.scored_nodes.insert(mid);
                if(gate.cur_topk) observed_topk.observe(mid_dist, is_deleted(mid));
                if(state_topk.observe(mid_dist, is_deleted(mid))) {
                    ++onehop_changes; ++stats.gate.onehop_topk_changes;
                }
                recordScoredEdge(stats.first_hop_edges, cur, i);

                const bool mid_admitted =
                    q->top_candidates.size() < q->L || q->get_dist_bound() > mid_dist;

                if(mid_admitted) {
                    // In bound-two-hop mode direct_mid means "baseline beam
                    // admitted this mid" (not mid_dist <= cur_dist).
                    ++stats.direct_mid_count;
                    q->push(mid, mid_dist, is_deleted(mid));
                    if(stats.collect_search_state) trace_admission(mid,mid_dist,trace_depth[cur]+1);
                    recordAdmittedEdge(stats.first_hop_edges, cur, i);
                    admitted_origin[mid] = {cur, i, false};
                    auto broot = stats.bridge_root_by_node.find(cur);
                    if(broot != stats.bridge_root_by_node.end()) {
                        stats.bridge_root_by_node[mid] = broot->second;
                    }
                    ++stats.mid_enqueue_count;
                    continue;
                }

                // Gate BEFORE touching N(mid) or computing any nn distance.
                // A skipped probe never marks its unscored nn nodes visited.
                ++stats.gate.rejected_mids;
                if(deferred) {
                    pending.push_back({mid, mid_dist, i, static_cast<unsigned>(getSearchEdgeType(cur, i))});
                } else if(allows_mid(mid, mid_dist, i)) {
                    probe_mid(mid, mid_dist, i);
                }
                // IMPORTANT: no fallback enqueue of mid. Baseline rejected it,
                // so keeping it rejected isolates the effect of bridge expansion.
            }
            if(deferred) {
                progress.observe(onehop_changes, state_topk.full());
                std::vector<PendingJumpMid> eligible;
                for(const auto& m : pending) {
                    if(allows_mid(m.id, m.distance, m.neighbor_index)) eligible.push_back(m);
                }
                stats.gate.eligible_mids += eligible.size();
                if(!eligible.empty()) {
                    ++stats.gate.trigger_checks;
                    const bool stall = progress.stalled();
                    // Exact next-iteration stop test, evaluated after ordinary one-hop.
                    const bool frontier = q->is_empty() ||
                        -q->candidate_set.top().first > q->get_dist_bound();
                    const bool fire = gate.trigger == JumpTrigger::ALWAYS ||
                        (state_topk.full() && (stall ||
                         (gate.trigger == JumpTrigger::STALL_FRONTIER && frontier)));
                    if(!fire) stats.gate.blocked_trigger += eligible.size();
                    else {
                        ++stats.gate.trigger_fires;
                        if(gate.trigger != JumpTrigger::ALWAYS) {
                            if(stall) ++stats.gate.stall_fires;
                            if(gate.trigger == JumpTrigger::STALL_FRONTIER && frontier)
                                ++stats.gate.frontier_fires;
                        }
                        const auto chosen = selectJumpMids(eligible, gate.selection, gate.max_mids_per_cur);
                        stats.gate.selected_mids += chosen.size();
                        stats.gate.blocked_selection += eligible.size() - chosen.size();
                        for(size_t index : chosen) {
                            const auto& m = eligible[index];
                            // Earlier probes may have tightened bound or used the budget.
                            if(allows_mid(m.id, m.distance, m.neighbor_index))
                                probe_mid(m.id, m.distance, m.neighbor_index);
                        }
                    }
                }
            }
        }

        q->releaseVisitedList();
        return q->get_result(k);
    }


    #include "hnsw_ngfix_clean32_impl.h"
    #include "hnsw_ngfix_selective_prefix_impl.h"
    #include "hnsw_ngfix_cur_mid_full_impl.h"
    #include "hnsw_ngfix_unbounded_clean_impl.h"

    // ============================================================
    // History-gated jump search.
    // Only a popped cur that is PRESENT in history_frequency and whose
    // frequency >= history_threshold may execute the two-hop jump logic.
    // All other cur expansions are exact one-hop greedy fallbacks.
    // ============================================================
    std::vector<std::pair<float, id_t> >
    searchKnnHistoryTwoHop(T* query_data, size_t k, size_t ef, size_t& ndc,
                           TwoHopSearchStats* stats_ptr = nullptr)
    {
        TwoHopSearchStats local_stats;
        TwoHopSearchStats& stats = stats_ptr ? *stats_ptr : local_stats;

        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        enum NodeState : uint8_t { UNSEEN = 0, SCORED = 1, ENQUEUED = 2 };
        std::vector<uint8_t> state(max_elements, UNSEEN);
        std::vector<float> dist_cache(max_elements, 0.0f);

        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        const float ep_dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        dist_cache[entry_point] = ep_dist;
        state[entry_point] = ENQUEUED;
        q->push(entry_point, ep_dist, is_deleted(entry_point));
        q->set_visited(entry_point);

        auto score_node = [&](id_t x, bool is_mid, bool* newly_computed = nullptr) -> float {
            bool did_compute = false;
            if(state[x] == UNSEEN) {
                dist_cache[x] = getQueryDist(x, query_data);
                state[x] = SCORED;
                ++ndc;
                ++stats.distance_cmps;
                if(is_mid) ++stats.mid_distance_cmps;
                else ++stats.nn_distance_cmps;
                did_compute = true;
            } else {
                ++stats.cache_hits;
            }
            if(newly_computed) *newly_computed = did_compute;
            return dist_cache[x];
        };

        auto enqueue_node = [&](id_t x, float d) -> bool {
            if(state[x] == ENQUEUED) return false;
            state[x] = ENQUEUED;
            q->set_visited(x);
            q->push(x, d, is_deleted(x));
            return true;
        };

        while(!q->is_empty()) {
            if(hop > hop_limit) {
                ++stats.hop_limit_stop_count;
                break;
            }

            auto current_node_pair = q->get_next_id();
            const id_t cur = current_node_pair.second;
            const float cur_dist = -current_node_pair.first;
            if(cur_dist > q->get_dist_bound()) break;

            ++hop;
            ++stats.cur_pop_count;

            const bool allow_twohop =
                enable_history_twohop &&
                cur < history_frequency.size() &&
                history_frequency[cur] >= 0.0f &&
                history_frequency[cur] >= history_threshold;

            if(allow_twohop) ++stats.history_allowed_cur_count;
            else ++stats.history_fallback_cur_count;

            std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);

            for(int i = st; i < st + sz; ++i) {
                const id_t mid = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i + 1]);
                    __builtin_prefetch(getData(outs[i + 1]), 0, 1);
                }

                if(state[mid] == ENQUEUED) {
                    ++stats.cache_hits;
                    ++stats.skipped_mid_count;
                    continue;
                }

                const float mid_dist = score_node(mid, true);

                if(!allow_twohop || mid_dist <= cur_dist) {
                    ++stats.direct_mid_count;
                    if(enqueue_node(mid, mid_dist)) {
                        ++stats.mid_enqueue_count;
                        recordEnqueuedEdge(stats.first_hop_edges, cur, i);
                    }
                    continue;
                }

                ++stats.twohop_mid_scan_count;
                bool has_jump = false;
                struct JumpCandidate {
                    id_t id;
                    float dist;
                    int neighbor_index;
                };
                std::vector<JumpCandidate> jump_candidates;
                size_t fresh_nn_distance_cmps_for_mid = 0;

                std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
                auto [nn_outs, nn_sz, nn_st] = getNeighbors(mid);
                jump_candidates.reserve(nn_sz);

                for(int j = nn_st; j < nn_st + nn_sz; ++j) {
                    const id_t nn = nn_outs[j];
                    ++stats.checked_nn_count;
                    recordExaminedEdge(stats.second_hop_edges, mid, j);
                    const SearchEdgeType first_edge_type = getSearchEdgeType(cur, i);
                    const SearchEdgeType second_edge_type = getSearchEdgeType(mid, j);
                    recordEdgePair(stats.twohop_examined_pairs, first_edge_type, second_edge_type);

                    if(j < nn_st + nn_sz - 1) {
                        q->prefetch_visited_list(nn_outs[j + 1]);
                        __builtin_prefetch(getData(nn_outs[j + 1]), 0, 1);
                    }

                    if(state[nn] == ENQUEUED) {
                        ++stats.cache_hits;
                        continue;
                    }

                    bool nn_computed = false;
                    const float nn_dist = score_node(nn, false, &nn_computed);
                    if(nn_computed) {
                        ++fresh_nn_distance_cmps_for_mid;
                    }
                    if(nn_dist < cur_dist) {
                        has_jump = true;
                        ++stats.jump_candidate_count;
                        recordEdgePair(stats.jump_pairs, first_edge_type, second_edge_type);
                        jump_candidates.push_back({nn, nn_dist, j});
                    }
                }

                if(has_jump) {
                    ++stats.jump_mid_count;
                    stats.nn_jump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    std::sort(jump_candidates.begin(), jump_candidates.end(),
                              [](const JumpCandidate& a, const JumpCandidate& b) {
                                  return a.dist < b.dist;
                              });
                    for(const auto& jc : jump_candidates) {
                        if(enqueue_node(jc.id, jc.dist)) {
                            ++stats.nn_enqueue_count;
                            recordEnqueuedEdge(stats.second_hop_edges, mid, jc.neighbor_index);
                        }
                    }
                } else {
                    ++stats.failed_jump_mid_count;
                    stats.nn_nojump_distance_cmps += fresh_nn_distance_cmps_for_mid;
                    if(enqueue_node(mid, mid_dist)) {
                        ++stats.mid_enqueue_count;
                        recordEnqueuedEdge(stats.first_hop_edges, cur, i);
                    }
                }
            }
        }

        q->releaseVisitedList();
        return q->get_result(k);
    }


    // ============================================================
    // Two-hop expansion ablation: every popped cur scans cur -> mid -> nn,
    // and BOTH mid and nn are allowed to enter the candidate heap.
    // This is intentionally not a jump algorithm; it measures the effect of
    // simply widening the frontier to two hops while retaining cache/prefetch.
    // The RoarGraph jump condition is still counted observationally.
    // ============================================================
    std::vector<std::pair<float, id_t> >
    searchKnnTwoHopExpansion(T* query_data, size_t k, size_t ef, size_t& ndc,
                             TwoHopSearchStats* stats_ptr = nullptr)
    {
        TwoHopSearchStats local_stats;
        TwoHopSearchStats& stats = stats_ptr ? *stats_ptr : local_stats;

        size_t hop_limit = std::numeric_limits<int>::max();
        size_t hop = 0;
        if(ef < k) {
            hop_limit = ef;
            ef = k;
        }

        enum NodeState : uint8_t { UNSEEN = 0, SCORED = 1, ENQUEUED = 2 };
        std::vector<uint8_t> state(max_elements, UNSEEN);
        std::vector<float> dist_cache(max_elements, 0.0f);

        Search_QuadHeap q0(ef, visited_list_pool_);
        auto q = &q0;

        const float ep_dist = getQueryDist(entry_point, query_data);
        ++ndc;
        ++stats.distance_cmps;
        ++stats.entry_distance_cmps;
        dist_cache[entry_point] = ep_dist;
        state[entry_point] = ENQUEUED;
        q->push(entry_point, ep_dist, is_deleted(entry_point));
        q->set_visited(entry_point);

        auto score_node = [&](id_t x, bool is_mid) -> float {
            if(state[x] == UNSEEN) {
                dist_cache[x] = getQueryDist(x, query_data);
                state[x] = SCORED;
                ++ndc;
                ++stats.distance_cmps;
                if(is_mid) ++stats.mid_distance_cmps;
                else ++stats.nn_distance_cmps;
            } else {
                ++stats.cache_hits;
            }
            return dist_cache[x];
        };

        auto enqueue_node = [&](id_t x, float d) -> bool {
            if(state[x] == ENQUEUED) return false;
            state[x] = ENQUEUED;
            q->set_visited(x);
            q->push(x, d, is_deleted(x));
            return true;
        };

        while(!q->is_empty()) {
            if(hop > hop_limit) {
                ++stats.hop_limit_stop_count;
                break;
            }

            auto current_node_pair = q->get_next_id();
            const id_t cur = current_node_pair.second;
            const float cur_dist = -current_node_pair.first;
            if(cur_dist > q->get_dist_bound()) break;

            ++hop;
            ++stats.cur_pop_count;

            std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
            auto [outs, sz, st] = getNeighbors(cur);

            for(int i = st; i < st + sz; ++i) {
                const id_t mid = outs[i];
                ++stats.onehop_mid_examined;
                recordExaminedEdge(stats.first_hop_edges, cur, i);

                if(i < st + sz - 1) {
                    q->prefetch_visited_list(outs[i + 1]);
                    __builtin_prefetch(getData(outs[i + 1]), 0, 1);
                }

                const float mid_dist = score_node(mid, true);
                if(enqueue_node(mid, mid_dist)) {
                    ++stats.mid_enqueue_count;
                    recordEnqueuedEdge(stats.first_hop_edges, cur, i);
                }

                ++stats.twohop_mid_scan_count;
                std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
                auto [nn_outs, nn_sz, nn_st] = getNeighbors(mid);

                for(int j = nn_st; j < nn_st + nn_sz; ++j) {
                    const id_t nn = nn_outs[j];
                    ++stats.checked_nn_count;
                    recordExaminedEdge(stats.second_hop_edges, mid, j);
                    const SearchEdgeType first_edge_type = getSearchEdgeType(cur, i);
                    const SearchEdgeType second_edge_type = getSearchEdgeType(mid, j);
                    recordEdgePair(stats.twohop_examined_pairs, first_edge_type, second_edge_type);

                    if(j < nn_st + nn_sz - 1) {
                        q->prefetch_visited_list(nn_outs[j + 1]);
                        __builtin_prefetch(getData(nn_outs[j + 1]), 0, 1);
                    }

                    const float nn_dist = score_node(nn, false);

                    // Observational only: count how many scanned two-hop paths would
                    // qualify as a RoarGraph jump. It does not affect enqueueing.
                    if(mid_dist > cur_dist && nn_dist < cur_dist) {
                        ++stats.jump_candidate_count;
                        recordEdgePair(stats.jump_pairs, first_edge_type, second_edge_type);
                    }

                    if(enqueue_node(nn, nn_dist)) {
                        ++stats.nn_enqueue_count;
                        recordEnqueuedEdge(stats.second_hop_edges, mid, j);
                    }
                }
            }
        }

        q->releaseVisitedList();
        return q->get_result(k);
    }


    void printGraphEdgeStats(const GraphEdgeStats& edge_stats) const {
        std::cout << "Graph directed-edge statistics:\n";
        std::cout << "  total: " << edge_stats.total_edges << "\n";
        std::cout << "  HNSW_BASE: " << edge_stats.hnsw_base_edges << "\n";
        std::cout << "  repair area (NGFix + RFix): " << edge_stats.repair_area_edges << "\n";
        std::cout << "    NGFIX: " << edge_stats.ngfix_edges << "\n";
        std::cout << "      finite EH: " << edge_stats.ngfix_finite_eh_edges << "\n";
        std::cout << "      EH_INF: " << edge_stats.ngfix_eh_inf_edges << "\n";
        std::cout << "    RFIX (eh == MAX_S + 1 == "
                  << static_cast<unsigned>(RFIX_EH_MARKER) << "): "
                  << edge_stats.rfix_edges << "\n";
        std::cout << "    NGFix exact EH histogram (non-zero bins):\n";
        for(size_t eh = 0; eh <= MAX_S; ++eh) {
            if(edge_stats.ngfix_eh_hist[eh] != 0) {
                std::cout << "      EH=" << eh << ": "
                          << edge_stats.ngfix_eh_hist[eh] << "\n";
            }
        }
        if(edge_stats.ngfix_eh_hist[EH_HIST_INF_INDEX] != 0) {
            std::cout << "      EH=INF: "
                      << edge_stats.ngfix_eh_hist[EH_HIST_INF_INDEX] << "\n";
        }
    }

    void writeGraphEhDistribution(std::ostream& out) const {
        const auto edge_stats = getGraphEdgeStats();
        out << "eh,count,ratio\n";
        const double denom = edge_stats.ngfix_edges > 0
            ? static_cast<double>(edge_stats.ngfix_edges) : 1.0;
        for(size_t eh = 0; eh <= MAX_S; ++eh) {
            const size_t c = edge_stats.ngfix_eh_hist[eh];
            if(c != 0) out << eh << "," << c << "," << (c / denom) << "\n";
        }
        const size_t infc = edge_stats.ngfix_eh_hist[EH_HIST_INF_INDEX];
        if(infc != 0) out << "INF," << infc << "," << (infc / denom) << "\n";
    }

    void printGraphInfo() {
        double avg_outdegree = 0;
        double avg_capacity = 0;

        std::cout << "current number of elements: " << n << "\n";
        std::cout << "max number of elements: " << max_elements << "\n";

        for(int i = 0; i < n; ++i) {
            avg_outdegree += GET_SZ((uint8_t*)Graph[i].neighbors);
            avg_capacity += GET_CAPACITY((uint8_t*)Graph[i].neighbors);
        }
        if(n > 0) {
            avg_outdegree /= n;
            avg_capacity /= n;
        }
        std::cout << "Average out-degree: " << avg_outdegree << "\n";
        std::cout << "Average Capacity: " << avg_capacity << "\n";
        std::cout << "entry point: " << entry_point << "\n";

        printGraphEdgeStats(getGraphEdgeStats());
    }

    void printEdgeUsageStats(const char* label, const EdgeUsageStats& e) const {
        std::cout << label << " examined total/HNSW/NGFix/RFix: "
                  << e.examined_total << "/"
                  << e.examined_hnsw_base << "/"
                  << e.examined_ngfix << "/"
                  << e.examined_rfix << "\n";
        std::cout << label << " examined NGFix finite-EH/EH_INF: "
                  << e.examined_ngfix_finite_eh << "/"
                  << e.examined_ngfix_eh_inf << "\n";
        std::cout << label << " enqueued total/HNSW/NGFix/RFix: "
                  << e.enqueued_total << "/"
                  << e.enqueued_hnsw_base << "/"
                  << e.enqueued_ngfix << "/"
                  << e.enqueued_rfix << "\n";
        std::cout << label << " enqueued NGFix finite-EH/EH_INF: "
                  << e.enqueued_ngfix_finite_eh << "/"
                  << e.enqueued_ngfix_eh_inf << "\n";
    }

    void printEdgePairStats(const char* label, const EdgePairStats& p) const {
        std::cout << label << " total: " << p.total << "\n";
        std::cout << "  HNSW->HNSW / HNSW->NGFix / HNSW->RFix: "
                  << p.hnsw_hnsw << " / " << p.hnsw_ngfix << " / " << p.hnsw_rfix << "\n";
        std::cout << "  NGFix->HNSW / NGFix->NGFix / NGFix->RFix: "
                  << p.ngfix_hnsw << " / " << p.ngfix_ngfix << " / " << p.ngfix_rfix << "\n";
        std::cout << "  RFix->HNSW / RFix->NGFix / RFix->RFix: "
                  << p.rfix_hnsw << " / " << p.rfix_ngfix << " / " << p.rfix_rfix << "\n";
    }

    void printSearchStats(const TwoHopSearchStats& stats) const {
        std::cout << "NDC (excluding entry point): " << stats.distance_cmps << "\n";
        std::cout << "  entry distance cmps (included in NDC): " << stats.entry_distance_cmps << "\n";
        std::cout << "  mid distance cmps: " << stats.mid_distance_cmps << "\n";
        std::cout << "  nn distance cmps: " << stats.nn_distance_cmps << "\n";
        std::cout << "    nn cmps under successful-jump mids: " << stats.nn_jump_distance_cmps << "\n";
        std::cout << "    nn cmps under failed-jump mids: " << stats.nn_nojump_distance_cmps << "\n";
        std::cout << "  cache hits: " << stats.cache_hits << "\n";
        std::cout << "cur pops: " << stats.cur_pop_count << "\n";
        std::cout << "hop-limit stops: " << stats.hop_limit_stop_count << "\n";
        std::cout << "one-hop mids examined: " << stats.onehop_mid_examined << "\n";
        std::cout << "two-hop mids scanned: " << stats.twohop_mid_scan_count << "\n";
        std::cout << "two-hop nn checked: " << stats.checked_nn_count << "\n";
        std::cout << "direct mids: " << stats.direct_mid_count << "\n";
        std::cout << "jump mids: " << stats.jump_mid_count << "\n";
        std::cout << "failed-jump mids: " << stats.failed_jump_mid_count << "\n";
        std::cout << "skipped already-enqueued mids: " << stats.skipped_mid_count << "\n";
        std::cout << "jump candidates: " << stats.jump_candidate_count << "\n";
        std::cout << "mid enqueues: " << stats.mid_enqueue_count << "\n";
        std::cout << "nn enqueues: " << stats.nn_enqueue_count << "\n";
        std::cout << "history two-hop cur: " << stats.history_allowed_cur_count << "\n";
        std::cout << "history fallback cur: " << stats.history_fallback_cur_count << "\n";

        printEdgeUsageStats("first-hop edges", stats.first_hop_edges);
        printEdgeUsageStats("second-hop edges", stats.second_hop_edges);
        printEdgePairStats("all scanned two-hop edge pairs", stats.twohop_examined_pairs);
        printEdgePairStats("jump-qualified two-hop edge pairs", stats.jump_pairs);
    }

};

}
