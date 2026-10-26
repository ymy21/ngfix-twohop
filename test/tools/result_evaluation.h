#pragma once

#include <vector>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <array>
#include <sys/time.h>
#include <unordered_set>
#include <unordered_map>

#include "../../ngfixlib/graph/hnsw_ngfix.h"

using namespace ngfixlib;

struct SearchResult
{
    BoundGateStats gate;
    float recall = 0.0f;
    size_t ndc = 0;
    int64_t latency_us = 0;
    double qps = 0.0;
    double rderr = 0.0;

    // Search-time NDC breakdown. Entry point IS included in total NDC.
    // ndc = ndc_entry + ndc_mid + ndc_nn
    // ndc_nn = ndc_nn_jump + ndc_nn_nojump
    // nn_jump / nn_nojump are attributed by the FINAL outcome of each mid:
    // all fresh NN computations under a successful-jump mid go to nn_jump;
    // all fresh NN computations under a failed-jump mid go to nn_nojump.
    size_t ndc_entry = 0;
    size_t ndc_mid = 0;
    size_t ndc_nn = 0;
    size_t ndc_nn_jump = 0;
    size_t ndc_nn_nojump = 0;

    // Number of cur nodes popped from the candidate queue and actually expanded.
    size_t hops = 0;

    // Mid / second-hop traversal counters.
    size_t mid_examined = 0;
    size_t twohop_mid_scans = 0;
    size_t nn_checked = 0;

    // Mid classification. For normal twohop mode, every non-skipped mid is one of:
    // direct (mid_dist <= cur_dist), jump, or failed_jump.
    size_t direct_mids = 0;
    size_t jump_mids = 0;
    size_t failed_jump_mids = 0;
    size_t skipped_mids = 0;
    size_t skipped_mid_from_success_probe = 0;
    size_t skipped_mid_from_failed_probe = 0;
    size_t skipped_mid_from_admitted_jump = 0;
    size_t jump_candidates = 0;
    size_t jump_candidates_gt = 0;
    size_t final_gt_direct_jump = 0;
    size_t final_gt_bridge_descendant = 0;
    size_t final_gt_non_gt_bridge_root = 0;
    size_t final_topk_bridge_descendant = 0;

    // Paired baseline-vs-boundtwohop diagnostics (populated by
    // search_mode=boundtwohopcmp). The baseline pass is excluded from the
    // reported boundtwohop latency/NDC; it is only used to identify which GTs
    // are genuinely recovered or lost by the bridge search.
    float paired_baseline_recall = 0.0f;
    size_t paired_baseline_ndc = 0;
    size_t recovered_gt_total = 0;
    size_t recovered_gt_direct_jump = 0;
    size_t recovered_gt_bridge_descendant = 0;
    size_t recovered_gt_non_gt_bridge_root = 0;
    size_t lost_gt_total = 0;

    // Two-hop edge-type pair counters. These are especially important for
    // boundtwohop: examined_* is the opportunity denominator, while jump_*
    // counts pairs whose second-hop nn was actually admitted into the beam.
    size_t examined_pair_hh = 0, examined_pair_hn = 0, examined_pair_hr = 0;
    size_t examined_pair_nh = 0, examined_pair_nn = 0, examined_pair_nr = 0;
    size_t examined_pair_rh = 0, examined_pair_rn = 0, examined_pair_rr = 0;
    size_t scored_pair_hh = 0, scored_pair_hn = 0, scored_pair_hr = 0;
    size_t scored_pair_nh = 0, scored_pair_nn = 0, scored_pair_nr = 0;
    size_t scored_pair_rh = 0, scored_pair_rn = 0, scored_pair_rr = 0;
    size_t jump_pair_hh = 0, jump_pair_hn = 0, jump_pair_hr = 0;
    size_t jump_pair_nh = 0, jump_pair_nn = 0, jump_pair_nr = 0;
    size_t jump_pair_rh = 0, jump_pair_rn = 0, jump_pair_rr = 0;

    // Traversed/examined directed edges. An examined edge is an adjacency entry
    // scanned by the search, regardless of whether its target is already visited.
    size_t edges_total = 0;
    size_t edges_hnsw = 0;
    size_t edges_ngfix = 0;
    size_t edges_rfix = 0;

    size_t first_edges_total = 0;
    size_t first_edges_hnsw = 0;
    size_t first_edges_ngfix = 0;
    size_t first_edges_rfix = 0;

    size_t second_edges_total = 0;
    size_t second_edges_hnsw = 0;
    size_t second_edges_ngfix = 0;
    size_t second_edges_rfix = 0;

    // Baseline measurement-correctness stages. These are aggregated over all
    // traversed search edges (first + second hop; baseline has first hop only).
    size_t examined_total_m = 0, examined_hnsw_m = 0, examined_ngfix_m = 0, examined_rfix_m = 0;
    size_t examined_ngfix_finite_m = 0, examined_ngfix_inf_m = 0;
    size_t scored_total_m = 0, scored_hnsw_m = 0, scored_ngfix_m = 0, scored_rfix_m = 0;
    size_t scored_ngfix_finite_m = 0, scored_ngfix_inf_m = 0;
    size_t admitted_total_m = 0, admitted_hnsw_m = 0, admitted_ngfix_m = 0, admitted_rfix_m = 0;
    size_t admitted_ngfix_finite_m = 0, admitted_ngfix_inf_m = 0;
    size_t expanded_total_m = 0, expanded_hnsw_m = 0, expanded_ngfix_m = 0, expanded_rfix_m = 0;
    size_t expanded_ngfix_finite_m = 0, expanded_ngfix_inf_m = 0;

    std::array<size_t, MAX_S + 2> examined_ngfix_eh_hist{};
    std::array<size_t, MAX_S + 2> scored_ngfix_eh_hist{};
    std::array<size_t, MAX_S + 2> admitted_ngfix_eh_hist{};
    std::array<size_t, MAX_S + 2> expanded_ngfix_eh_hist{};
};

// Query-level EH buckets chosen to match this repository's current builder:
// NGFix(q,gt,10,10) creates finite defects mainly at EH 11..19;
// NGFix(q,gt,100,100) creates finite defects mainly at EH 101..199.
// The "other" ranges are retained as sanity checks rather than silently dropped.
static constexpr size_t QUERY_EH_BUCKETS = 8;
static const char* QUERY_EH_BUCKET_NAMES[QUERY_EH_BUCKETS] = {
    "eh_0_10", "eh_11_19", "eh_20_100", "eh_101_125",
    "eh_126_150", "eh_151_175", "eh_176_200", "eh_inf"
};

static std::array<size_t, QUERY_EH_BUCKETS>
BucketEhHist(const std::array<size_t, MAX_S + 2>& hist) {
    std::array<size_t, QUERY_EH_BUCKETS> b{};
    for(size_t eh = 0; eh <= MAX_S; ++eh) {
        const size_t c = hist[eh];
        if(eh <= 10) b[0] += c;
        else if(eh <= 19) b[1] += c;
        else if(eh <= 100) b[2] += c;
        else if(eh <= 125) b[3] += c;
        else if(eh <= 150) b[4] += c;
        else if(eh <= 175) b[5] += c;
        else b[6] += c;
    }
    b[7] = hist[MAX_S + 1];
    return b;
}


void AllQueriesEvaluation(
    const std::vector<SearchResult>& results,
    double& avg_recall,
    double& avg_ndc,
    double& avg_latency_ms,
    double& avg_rderr,
    double& aggregate_qps)
{
    const size_t test_number = results.size();
    if(test_number == 0) return;

    double total_latency_us = 0.0;

    for(const auto &r:results){
        avg_recall += r.recall;
        avg_ndc += static_cast<double>(r.ndc);
        total_latency_us += static_cast<double>(r.latency_us);
        avg_rderr += r.rderr;
    }

    avg_recall /= test_number;
    avg_ndc /= test_number;
    avg_latency_ms = total_latency_us / test_number / 1000.0;
    avg_rderr /= test_number;
    aggregate_qps = total_latency_us > 0.0
        ? static_cast<double>(test_number) * 1000000.0 / total_latency_us
        : 0.0;

    std::cerr << "average recall: " << avg_recall << "\n";
    std::cerr << "average latency: " << avg_latency_ms << "ms\n";
    std::cerr << "average distance computation: " << avg_ndc << "\n";
    std::cerr << "aggregate qps: " << aggregate_qps << "\n";
}


template<typename T>
SearchResult TestSingleQuery(
    T* query_data,
    int* gt,
    size_t k,
    size_t efs,
    float dabs_gamma,
    typename HNSW_NGFix<T>::DABSDistanceTransform dabs_transform,
    size_t query_id,
    HNSW_NGFix<T>* searcher,
    const std::string& mode,
    std::ostream* jump_event_out = nullptr,
    std::ostream* jump_mid_out = nullptr,
    std::ostream* jump_hopbin_out = nullptr,
    bool event_trace_compact = false,
    std::ostream* nn_out = nullptr,
    std::ostream* search_state_out = nullptr)
{
    size_t ndc = 0;
    typename HNSW_NGFix<T>::TwoHopSearchStats stats;
    const bool collect_event_trace =
        (jump_event_out != nullptr || jump_mid_out != nullptr || jump_hopbin_out != nullptr || nn_out != nullptr || search_state_out != nullptr);
    stats.collect_search_state = search_state_out != nullptr;
    stats.collect_nn_events = nn_out != nullptr;
    stats.collect_jump_events = collect_event_trace;
    stats.collect_node_ids = collect_event_trace;

    std::unordered_set<id_t> paired_baseline_gt_results;
    std::unordered_set<id_t> paired_baseline_scored_nodes;
    std::unordered_set<id_t> paired_baseline_expanded_nodes;
    size_t paired_baseline_ndc = 0;

    if(mode == "boundtwohopcmp") {
        typename HNSW_NGFix<T>::TwoHopSearchStats baseline_stats;
        baseline_stats.collect_node_ids = collect_event_trace;
        auto baseline = searcher->searchKnnWithStats(
            query_data, k, efs, paired_baseline_ndc, baseline_stats);
        if(collect_event_trace) {
            paired_baseline_scored_nodes = std::move(baseline_stats.scored_nodes);
            paired_baseline_expanded_nodes = std::move(baseline_stats.expanded_nodes);
        }
        std::unordered_set<id_t> gt_lookup_tmp;
        for(size_t i = 0; i < k; ++i) gt_lookup_tmp.insert(gt[i]);
        for(size_t i = 0; i < baseline.size() && i < k; ++i) {
            if(gt_lookup_tmp.count(baseline[i].second))
                paired_baseline_gt_results.insert(baseline[i].second);
        }
    }

    timeval val;
    gettimeofday(&val,NULL);

    std::vector<std::pair<float,id_t>> aknns;

    if(mode == "twohop")
    {
        aknns = searcher->searchKnnTwoHop(
            query_data, k, efs, ndc, &stats);
    }
    else if(mode == "boundtwohop" || mode == "boundtwohopcmp")
    {
        aknns = searcher->searchKnnBoundTwoHop(
            query_data, k, efs, ndc, &stats);
    }
    else if(mode == "history")
    {
        aknns = searcher->searchKnnHistoryTwoHop(
            query_data, k, efs, ndc, &stats);
    }
    else if(mode == "dabs")
    {
        aknns = searcher->searchKnnDABSWithStats(
            query_data, k, dabs_gamma, dabs_transform, ndc, stats);
    }
    else
    {
        // Important: use the observation-only baseline implementation so that
        // baseline gets the same edge/NDC statistics as two-hop search.
        aknns = searcher->searchKnnWithStats(
            query_data, k, efs, ndc, stats);
    }

    timeval newVal;
    gettimeofday(&newVal,NULL);

    if(aknns.size() < k) {
        throw std::runtime_error(
            "Search returned fewer than K live results; check the index and K.");
    }

    const int64_t diff =
        (newVal.tv_sec*1000000LL + newVal.tv_usec)
        - (val.tv_sec*1000000LL + val.tv_usec);

    std::unordered_set<id_t> gtset;
    for(size_t i = 0; i < k; i++)
        gtset.insert(gt[i]);
    const std::unordered_set<id_t> gt_lookup = gtset;

    int acc = 0;
    for(size_t i = 0; i < k; i++)
    {
        if(gtset.count(aknns[i].second))
        {
            acc++;
            gtset.erase(aknns[i].second);
        }
    }

    double rderr = 0.0;
    for(size_t i = 0; i < k; i++)
    {
        float d0 = searcher->getDist(aknns[i].second, query_data);
        float d1 = searcher->getDist(gt[i], query_data);

        if(fabs(d1) > 0.00001)
            rderr += d0/d1;
    }

    SearchResult result;
    result.gate = stats.gate;
    result.recall = static_cast<float>(acc) / k;
    result.ndc = ndc;
    result.latency_us = diff;
    result.qps = diff > 0 ? 1000000.0 / static_cast<double>(diff) : 0.0;
    result.rderr = rderr / k;

    result.ndc_entry = stats.entry_distance_cmps;
    result.ndc_mid = stats.mid_distance_cmps;
    result.ndc_nn = stats.nn_distance_cmps;
    result.ndc_nn_jump = stats.nn_jump_distance_cmps;
    result.ndc_nn_nojump = stats.nn_nojump_distance_cmps;

    result.hops = stats.cur_pop_count;
    result.mid_examined = stats.onehop_mid_examined;
    result.twohop_mid_scans = stats.twohop_mid_scan_count;
    result.nn_checked = stats.checked_nn_count;

    result.direct_mids = stats.direct_mid_count;
    result.jump_mids = stats.jump_mid_count;
    result.failed_jump_mids = stats.failed_jump_mid_count;
    result.skipped_mids = stats.skipped_mid_count;
    result.skipped_mid_from_success_probe = stats.skipped_mid_from_success_probe;
    result.skipped_mid_from_failed_probe = stats.skipped_mid_from_failed_probe;
    result.skipped_mid_from_admitted_jump = stats.skipped_mid_from_admitted_jump;
    result.jump_candidates = stats.jump_candidate_count;

    for(const id_t x : stats.jump_admitted_nodes) {
        if(gt_lookup.count(x)) ++result.jump_candidates_gt;
    }
    for(size_t i = 0; i < aknns.size() && i < k; ++i) {
        const id_t x = aknns[i].second;
        auto broot = stats.bridge_root_by_node.find(x);
        if(broot == stats.bridge_root_by_node.end()) continue;

        ++result.final_topk_bridge_descendant;
        if(!gt_lookup.count(x)) continue;

        if(stats.jump_admitted_nodes.count(x)) {
            ++result.final_gt_direct_jump;
        } else {
            ++result.final_gt_bridge_descendant;
        }

        if(!gt_lookup.count(broot->second)) {
            ++result.final_gt_non_gt_bridge_root;
        }
    }

    if(mode == "boundtwohopcmp") {
        result.paired_baseline_ndc = paired_baseline_ndc;
        result.paired_baseline_recall =
            static_cast<float>(paired_baseline_gt_results.size()) / static_cast<float>(k);

        std::unordered_set<id_t> bound_gt_results;
        for(size_t i = 0; i < aknns.size() && i < k; ++i) {
            const id_t x = aknns[i].second;
            if(gt_lookup.count(x)) bound_gt_results.insert(x);
        }

        for(const id_t x : bound_gt_results) {
            if(paired_baseline_gt_results.count(x)) continue;
            ++result.recovered_gt_total;

            if(stats.jump_admitted_nodes.count(x)) {
                ++result.recovered_gt_direct_jump;
                continue;
            }

            auto broot = stats.bridge_root_by_node.find(x);
            if(broot != stats.bridge_root_by_node.end()) {
                ++result.recovered_gt_bridge_descendant;
                if(!gt_lookup.count(broot->second)) {
                    ++result.recovered_gt_non_gt_bridge_root;
                }
            }
        }

        for(const id_t x : paired_baseline_gt_results) {
            if(!bound_gt_results.count(x)) ++result.lost_gt_total;
        }
    }

    // Optional event-level trace.  This is deliberately emitted after the
    // paired baseline and bound search both finish, so each admitted jump root
    // can be labeled by baseline visibility/finality and by its eventual
    // contribution to final/recovered GT descendants.
    if(collect_event_trace && mode == "boundtwohopcmp") {
        if(search_state_out) {
            for(const auto& ev:stats.bound_mid_probe_events) {
                *search_state_out<<query_id<<','<<efs<<','<<ev.probe_id<<','<<ev.cur<<','<<ev.mid<<','<<ev.hop<<',';
                HNSW_NGFix<T>::writeBoundSearchSnapshot(*search_state_out,ev.state);
                *search_state_out<<'\n';
            }
        }
        std::unordered_set<id_t> bound_final_nodes;
        std::unordered_set<id_t> bound_final_gt_results;
        std::unordered_map<id_t, int> gt_rank;
        gt_rank.reserve(k * 2 + 1);
        for(size_t i = 0; i < k; ++i) gt_rank[gt[i]] = static_cast<int>(i + 1);
        for(size_t i = 0; i < aknns.size() && i < k; ++i) {
            const id_t x = aknns[i].second;
            bound_final_nodes.insert(x);
            if(gt_lookup.count(x)) bound_final_gt_results.insert(x);
        }

        // Two downstream-attribution views:
        // 1) nearest_*: unique attribution to the most recent jump root before GT;
        // 2) ancestor_*: inclusive attribution to every earlier jump root on a
        //    nested-jump chain.  The latter is for "did this jump eventually help
        //    reach GT?" and must NOT be summed as unique recovered GT counts.
        std::unordered_map<id_t, size_t> root_final_gt_desc_nearest;
        std::unordered_map<id_t, size_t> root_recovered_gt_desc_nearest;
        std::unordered_map<id_t, size_t> root_final_gt_desc_ancestor;
        std::unordered_map<id_t, size_t> root_recovered_gt_desc_ancestor;
        for(const id_t x : bound_final_gt_results) {
            auto it = stats.bridge_root_by_node.find(x);
            if(it == stats.bridge_root_by_node.end()) continue;
            id_t root = it->second;
            const bool recovered = !paired_baseline_gt_results.count(x);

            // If x itself is a direct jump GT, it is not its own descendant,
            // but an earlier parent jump root may still have causally led to it.
            if(root != x) {
                ++root_final_gt_desc_nearest[root];
                if(recovered) ++root_recovered_gt_desc_nearest[root];
                ++root_final_gt_desc_ancestor[root];
                if(recovered) ++root_recovered_gt_desc_ancestor[root];
            }

            std::unordered_set<id_t> seen_roots;
            seen_roots.insert(root);
            auto pit = stats.jump_parent_root.find(root);
            while(pit != stats.jump_parent_root.end()) {
                const id_t parent = pit->second;
                if(!seen_roots.insert(parent).second) break;
                ++root_final_gt_desc_ancestor[parent];
                if(recovered) ++root_recovered_gt_desc_ancestor[parent];
                pit = stats.jump_parent_root.find(parent);
            }
        }

        if(nn_out) {
            for(const auto& ev:stats.bound_nn_events) {
                const bool direct=ev.admitted && bound_final_gt_results.count(ev.nn) && !paired_baseline_gt_results.count(ev.nn);
                const size_t descendants=ev.admitted ? root_recovered_gt_desc_nearest[ev.nn] : 0;
                *nn_out << query_id << "," << efs << "," << ev.probe_id << "," << ev.hop << ","
                    << ev.cur << "," << ev.mid << "," << ev.nn << "," << ev.adjacency_ordinal << ","
                    << ev.first_type << "," << ev.second_type << "," << ev.second_eh << ","
                    << ev.pool_size << "," << ev.scored_size << "," << ev.censored << ","
                    << ev.selected << "," << ev.scored << "," << ev.admitted << ","
                    << ev.exact_dist << "," << ev.cheap_score << "," << ev.random_key << ","
                    << ev.exact_rank << "," << direct << "," << descendants << ","
                    << (ev.scored ? int(direct || descendants>0) : -1) << "\n";
            }
        }

        struct MidOutcome {
            size_t admitted = 0;
            size_t direct_new_gt = 0;
            size_t direct_shared_gt = 0;
            size_t sanity_gt_not_final = 0;
            size_t non_gt_roots = 0;
            size_t useful_roots_nearest = 0;
            size_t useful_roots_ancestor = 0;
            size_t recovered_desc_gt_nearest = 0;
            size_t final_desc_gt_nearest = 0;
            size_t recovered_desc_gt_ancestor = 0;
            size_t final_desc_gt_ancestor = 0;
        };
        std::unordered_map<id_t, MidOutcome> mid_outcomes;
        mid_outcomes.reserve(stats.bound_mid_probe_events.size() * 2 + 1);

        std::unordered_map<id_t, typename HNSW_NGFix<T>::NodeLocalStructure> node_struct_cache;
        std::unordered_map<uint64_t, typename HNSW_NGFix<T>::PairLocalStructure> pair_struct_cache;
        node_struct_cache.reserve(stats.bound_jump_events.size() + stats.bound_mid_probe_events.size() + 16);
        pair_struct_cache.reserve(stats.bound_jump_events.size() * 3 + stats.bound_mid_probe_events.size() * 2 + 16);

        auto node_struct = [&](id_t x) -> const typename HNSW_NGFix<T>::NodeLocalStructure& {
            auto it = node_struct_cache.find(x);
            if(it == node_struct_cache.end())
                it = node_struct_cache.emplace(x, searcher->getNodeLocalStructure(x)).first;
            return it->second;
        };
        auto pair_struct = [&](id_t a, id_t b) -> const typename HNSW_NGFix<T>::PairLocalStructure& {
            const uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
            auto it = pair_struct_cache.find(key);
            if(it == pair_struct_cache.end())
                it = pair_struct_cache.emplace(key, searcher->getPairLocalStructure(a, b)).first;
            return it->second;
        };

        if(jump_event_out) {
            for(const auto& ev : stats.bound_jump_events) {
                const bool nn_is_gt = gt_lookup.count(ev.nn) != 0;
                const bool cur_is_gt = gt_lookup.count(ev.cur) != 0;
                const bool mid_is_gt = gt_lookup.count(ev.mid) != 0;
                const int nn_gt_rank = nn_is_gt ? gt_rank[ev.nn] : -1;
                const bool baseline_scored_nn = paired_baseline_scored_nodes.count(ev.nn) != 0;
                const bool baseline_expanded_nn = paired_baseline_expanded_nodes.count(ev.nn) != 0;
                const bool baseline_final_gt = nn_is_gt && paired_baseline_gt_results.count(ev.nn) != 0;
                const bool bound_final = bound_final_nodes.count(ev.nn) != 0;
                const bool bound_final_gt = nn_is_gt && bound_final_gt_results.count(ev.nn) != 0;
                const size_t final_desc_nearest = root_final_gt_desc_nearest[ev.nn];
                const size_t recovered_desc_nearest = root_recovered_gt_desc_nearest[ev.nn];
                const size_t final_desc_ancestor = root_final_gt_desc_ancestor[ev.nn];
                const size_t recovered_desc_ancestor = root_recovered_gt_desc_ancestor[ev.nn];

                std::string cls;
                if(nn_is_gt) {
                    // Primary direct-GT partition is baseline final hit vs miss.
                    // An admitted exact GT should normally remain in final top-k
                    // for ef>=k under the same metric; violation is a sanity case.
                    if(!bound_final_gt) cls = "SANITY_GT_ADMITTED_NOT_FINAL";
                    else if(baseline_final_gt) cls = "DIRECT_GT_SHARED";
                    else cls = "DIRECT_GT_NEW";
                } else {
                    // Keep the primary router taxonomy simple: did this jump root
                    // have any final GT descendant or not?  recovered_desc and
                    // shared-desc counts remain separate numeric attributes.
                    cls = (final_desc_ancestor > 0) ? "ROUTER_TO_GT" : "ROUTER_NO_GT";
                }

                auto& mo = mid_outcomes[ev.mid];
                ++mo.admitted;
                if(nn_is_gt) {
                    if(bound_final_gt && !baseline_final_gt) ++mo.direct_new_gt;
                    else if(bound_final_gt && baseline_final_gt) ++mo.direct_shared_gt;
                    else ++mo.sanity_gt_not_final;
                } else {
                    ++mo.non_gt_roots;
                }
                if((bound_final_gt && !baseline_final_gt) || recovered_desc_nearest > 0)
                    ++mo.useful_roots_nearest;
                if((bound_final_gt && !baseline_final_gt) || recovered_desc_ancestor > 0)
                    ++mo.useful_roots_ancestor;
                mo.recovered_desc_gt_nearest += recovered_desc_nearest;
                mo.final_desc_gt_nearest += final_desc_nearest;
                mo.recovered_desc_gt_ancestor += recovered_desc_ancestor;
                mo.final_desc_gt_ancestor += final_desc_ancestor;

                const double hop_frac = stats.cur_pop_count > 0
                    ? static_cast<double>(ev.hop) / static_cast<double>(stats.cur_pop_count)
                    : 0.0;
                const double mid_gap = static_cast<double>(ev.mid_dist) - ev.bound_at_mid_reject;
                const double nn_margin = static_cast<double>(ev.bound_before_nn_admit) - ev.nn_dist;
                const double hop_over_ef = efs > 0
                    ? static_cast<double>(ev.hop > 0 ? ev.hop - 1 : 0) / static_cast<double>(efs)
                    : 0.0;
                const auto& cs = node_struct(ev.cur);
                const auto& ms = node_struct(ev.mid);
                const auto& ns = node_struct(ev.nn);
                const auto& cm = pair_struct(ev.cur, ev.mid);
                const auto& mn = pair_struct(ev.mid, ev.nn);
                const auto& cn = pair_struct(ev.cur, ev.nn);

                *jump_event_out
                    << query_id << "," << efs << "," << stats.cur_pop_count << ","
                    << ev.hop << "," << hop_frac << ","
                    << ev.cur << "," << ev.mid << "," << ev.nn << ","
                    << cur_is_gt << "," << mid_is_gt << "," << nn_gt_rank << ","
                    << ev.cur_dist << "," << ev.mid_dist << "," << ev.nn_dist << ","
                    << ev.bound_at_mid_reject << "," << ev.bound_before_nn_admit << ","
                    << mid_gap << "," << nn_margin << ","
                    << static_cast<int>(ev.first_edge_type) << ","
                    << static_cast<int>(ev.second_edge_type) << ","
                    << ev.mid_checked_nn << "," << ev.mid_fresh_nn_cmps << ","
                    << baseline_scored_nn << "," << baseline_expanded_nn << ","
                    << nn_is_gt << "," << baseline_final_gt << ","
                    << bound_final << "," << bound_final_gt << ","
                    << final_desc_nearest << "," << recovered_desc_nearest << ","
                    << (final_desc_nearest >= recovered_desc_nearest ? final_desc_nearest - recovered_desc_nearest : 0) << ","
                    << final_desc_ancestor << "," << recovered_desc_ancestor << ","
                    << (final_desc_ancestor >= recovered_desc_ancestor ? final_desc_ancestor - recovered_desc_ancestor : 0) << ","
                    << cls << "," << hop_over_ef << ","
                    << static_cast<unsigned int>(ev.first_edge_eh) << ","
                    << static_cast<unsigned int>(ev.second_edge_eh) << ","
                    << ev.rank_in_fresh_nn_by_query_dist << ","
                    << cs.total_degree << "," << cs.hnsw_degree << "," << cs.ngfix_degree << "," << cs.rfix_degree << ","
                    << cs.repair_degree << "," << cs.repair_ratio << "," << cs.ngfix_finite_degree << "," << cs.ngfix_eh_inf_degree << ","
                    << ms.total_degree << "," << ms.hnsw_degree << "," << ms.ngfix_degree << "," << ms.rfix_degree << ","
                    << ms.repair_degree << "," << ms.repair_ratio << "," << ms.ngfix_finite_degree << "," << ms.ngfix_eh_inf_degree << ","
                    << ns.total_degree << "," << ns.hnsw_degree << "," << ns.ngfix_degree << "," << ns.rfix_degree << ","
                    << ns.repair_degree << "," << ns.repair_ratio << "," << ns.ngfix_finite_degree << "," << ns.ngfix_eh_inf_degree << ","
                    << cm.data_dist << "," << cm.b_to_a_edge_type << "," << cm.common_neighbors << "," << cm.union_neighbors << "," << cm.neighbor_jaccard << ","
                    << mn.data_dist << "," << mn.b_to_a_edge_type << "," << mn.common_neighbors << "," << mn.union_neighbors << "," << mn.neighbor_jaccard << ","
                    << cn.data_dist << "," << cn.a_to_b_edge_type << "," << cn.b_to_a_edge_type << ","
                    << cn.common_neighbors << "," << cn.union_neighbors << "," << cn.neighbor_jaccard << "\n";
            }
        } else {
            // Even when the candidate-level file is disabled, compute mid
            // outcomes for the mid/hop-bin summaries.
            for(const auto& ev : stats.bound_jump_events) {
                const bool nn_is_gt = gt_lookup.count(ev.nn) != 0;
                const bool baseline_final_gt = nn_is_gt && paired_baseline_gt_results.count(ev.nn) != 0;
                const bool bound_final_gt = nn_is_gt && bound_final_gt_results.count(ev.nn) != 0;
                const size_t final_desc_nearest = root_final_gt_desc_nearest[ev.nn];
                const size_t recovered_desc_nearest = root_recovered_gt_desc_nearest[ev.nn];
                const size_t final_desc_ancestor = root_final_gt_desc_ancestor[ev.nn];
                const size_t recovered_desc_ancestor = root_recovered_gt_desc_ancestor[ev.nn];
                auto& mo = mid_outcomes[ev.mid];
                ++mo.admitted;
                if(nn_is_gt) {
                    if(bound_final_gt && !baseline_final_gt) ++mo.direct_new_gt;
                    else if(bound_final_gt && baseline_final_gt) ++mo.direct_shared_gt;
                    else ++mo.sanity_gt_not_final;
                } else {
                    ++mo.non_gt_roots;
                }
                if((bound_final_gt && !baseline_final_gt) || recovered_desc_nearest > 0)
                    ++mo.useful_roots_nearest;
                if((bound_final_gt && !baseline_final_gt) || recovered_desc_ancestor > 0)
                    ++mo.useful_roots_ancestor;
                mo.recovered_desc_gt_nearest += recovered_desc_nearest;
                mo.final_desc_gt_nearest += final_desc_nearest;
                mo.recovered_desc_gt_ancestor += recovered_desc_ancestor;
                mo.final_desc_gt_ancestor += final_desc_ancestor;
            }
        }

        if(jump_mid_out) {
            for(const auto& ev : stats.bound_mid_probe_events) {
                const auto oit = mid_outcomes.find(ev.mid);
                const MidOutcome empty{};
                const MidOutcome& mo = (oit == mid_outcomes.end()) ? empty : oit->second;
                const bool success = ev.admitted_nn > 0;
                const bool useful_mid = mo.direct_new_gt > 0 || mo.recovered_desc_gt_nearest > 0;
                if(event_trace_compact) {
                    const auto features = makeJumpFeatures(ev.cur_dist, ev.mid_dist,
                        ev.bound_at_reject, ev.hop, std::max(efs, k),
                        static_cast<unsigned>(ev.first_edge_type), ev.first_edge_eh);
                    *jump_mid_out << query_id << "," << efs << "," << ev.cur << "," << ev.mid
                        << "," << ev.fresh_nn_cmps << "," << useful_mid
                        << "," << mo.direct_new_gt << "," << mo.recovered_desc_gt_nearest
                        << "," << ev.admitted_nn;
                    for(float feature : features) *jump_mid_out << "," << feature;
                    *jump_mid_out << "\n";
                    continue;
                }
                const bool baseline_scored_cur = paired_baseline_scored_nodes.count(ev.cur) != 0;
                const bool baseline_expanded_cur = paired_baseline_expanded_nodes.count(ev.cur) != 0;
                const bool baseline_scored_mid = paired_baseline_scored_nodes.count(ev.mid) != 0;
                const bool baseline_expanded_mid = paired_baseline_expanded_nodes.count(ev.mid) != 0;
                const double hop_frac = stats.cur_pop_count > 0
                    ? static_cast<double>(ev.hop) / static_cast<double>(stats.cur_pop_count)
                    : 0.0;
                const double mid_gap = static_cast<double>(ev.mid_dist) - ev.bound_at_reject;
                const double hop_over_ef = efs > 0
                    ? static_cast<double>(ev.hop > 0 ? ev.hop - 1 : 0) / static_cast<double>(efs)
                    : 0.0;
                const auto& cs = node_struct(ev.cur);
                const auto& ms = node_struct(ev.mid);
                const auto& cm = pair_struct(ev.cur, ev.mid);

                const bool has_best_scored = ev.best_scored_nn != std::numeric_limits<id_t>::max();
                bool best_scored_nn_is_gt = false;
                bool best_scored_baseline_scored = false;
                bool best_scored_baseline_final_gt = false;
                bool best_scored_bound_final_gt = false;
                typename HNSW_NGFix<T>::NodeLocalStructure bs_ns{};
                typename HNSW_NGFix<T>::PairLocalStructure bs_mn{};
                typename HNSW_NGFix<T>::PairLocalStructure bs_cn{};
                if(has_best_scored) {
                    best_scored_nn_is_gt = gt_lookup.count(ev.best_scored_nn) != 0;
                    best_scored_baseline_scored = paired_baseline_scored_nodes.count(ev.best_scored_nn) != 0;
                    best_scored_baseline_final_gt = best_scored_nn_is_gt && paired_baseline_gt_results.count(ev.best_scored_nn) != 0;
                    best_scored_bound_final_gt = best_scored_nn_is_gt && bound_final_gt_results.count(ev.best_scored_nn) != 0;
                    bs_ns = node_struct(ev.best_scored_nn);
                    bs_mn = pair_struct(ev.mid, ev.best_scored_nn);
                    bs_cn = pair_struct(ev.cur, ev.best_scored_nn);
                }

                *jump_mid_out
                    << query_id << "," << efs << "," << stats.cur_pop_count << ","
                    << ev.hop << "," << hop_frac << ","
                    << ev.cur << "," << ev.mid << ","
                    << ev.cur_dist << "," << ev.mid_dist << "," << ev.bound_at_reject << ","
                    << mid_gap << "," << static_cast<int>(ev.first_edge_type) << ","
                    << ev.checked_nn << "," << ev.fresh_nn_cmps << ","
                    << ev.admitted_nn << "," << success << "," << useful_mid << ","
                    << baseline_scored_cur << "," << baseline_expanded_cur << ","
                    << baseline_scored_mid << "," << baseline_expanded_mid << ","
                    << mo.direct_new_gt << "," << mo.direct_shared_gt << ","
                    << mo.sanity_gt_not_final << "," << mo.non_gt_roots << ","
                    << mo.useful_roots_nearest << "," << mo.useful_roots_ancestor << ","
                    << mo.final_desc_gt_nearest << "," << mo.recovered_desc_gt_nearest << ","
                    << mo.final_desc_gt_ancestor << "," << mo.recovered_desc_gt_ancestor << ",";
                if(ev.best_admitted_nn == std::numeric_limits<id_t>::max())
                    *jump_mid_out << -1 << ",nan,";
                else
                    *jump_mid_out << ev.best_admitted_nn << "," << ev.best_admitted_nn_dist << ",";

                *jump_mid_out
                    << hop_over_ef << "," << static_cast<unsigned int>(ev.first_edge_eh) << ","
                    << cs.total_degree << "," << cs.hnsw_degree << "," << cs.ngfix_degree << "," << cs.rfix_degree << ","
                    << cs.repair_degree << "," << cs.repair_ratio << "," << cs.ngfix_finite_degree << "," << cs.ngfix_eh_inf_degree << ","
                    << ms.total_degree << "," << ms.hnsw_degree << "," << ms.ngfix_degree << "," << ms.rfix_degree << ","
                    << ms.repair_degree << "," << ms.repair_ratio << "," << ms.ngfix_finite_degree << "," << ms.ngfix_eh_inf_degree << ","
                    << cm.data_dist << "," << cm.b_to_a_edge_type << "," << cm.common_neighbors << "," << cm.union_neighbors << "," << cm.neighbor_jaccard << ",";

                if(!has_best_scored) {
                    *jump_mid_out << -1 << ",nan,-1,-1,0,0,0,0,"
                                  << "0,0,0,0,0,0,0,0,"
                                  << "nan,-1,0,0,0,nan,-1,-1,0,0,0\n";
                } else {
                    *jump_mid_out
                        << ev.best_scored_nn << "," << ev.best_scored_nn_dist << ","
                        << static_cast<int>(ev.best_scored_second_edge_type) << ","
                        << static_cast<unsigned int>(ev.best_scored_second_edge_eh) << ","
                        << best_scored_nn_is_gt << "," << best_scored_baseline_scored << ","
                        << best_scored_baseline_final_gt << "," << best_scored_bound_final_gt << ","
                        << bs_ns.total_degree << "," << bs_ns.hnsw_degree << "," << bs_ns.ngfix_degree << "," << bs_ns.rfix_degree << ","
                        << bs_ns.repair_degree << "," << bs_ns.repair_ratio << "," << bs_ns.ngfix_finite_degree << "," << bs_ns.ngfix_eh_inf_degree << ","
                        << bs_mn.data_dist << "," << bs_mn.b_to_a_edge_type << "," << bs_mn.common_neighbors << "," << bs_mn.union_neighbors << "," << bs_mn.neighbor_jaccard << ","
                        << bs_cn.data_dist << "," << bs_cn.a_to_b_edge_type << "," << bs_cn.b_to_a_edge_type << ","
                        << bs_cn.common_neighbors << "," << bs_cn.union_neighbors << "," << bs_cn.neighbor_jaccard << "\n";
                }
            }
        }

        if(jump_hopbin_out) {
            static constexpr size_t BINS = 20;
            struct BinAgg {
                size_t scans = 0, successful_mids = 0, failed_mids = 0, useful_mids = 0;
                size_t fresh_nn_ndc = 0, admitted_nn = 0;
                size_t direct_new_gt = 0, direct_shared_gt = 0;
                size_t recovered_desc_gt = 0;
                double mid_gap_sum = 0.0;
                double bound_sum = 0.0;
                size_t first_h = 0, first_n = 0, first_r = 0;
            };
            std::array<BinAgg, BINS> bins{};
            for(const auto& ev : stats.bound_mid_probe_events) {
                const size_t denom = std::max<size_t>(stats.cur_pop_count, 1);
                const size_t h0 = ev.hop > 0 ? ev.hop - 1 : 0;
                const size_t b = std::min<size_t>(BINS - 1, (h0 * BINS) / denom);
                auto& ba = bins[b];
                ++ba.scans;
                ba.fresh_nn_ndc += ev.fresh_nn_cmps;
                ba.admitted_nn += ev.admitted_nn;
                ba.mid_gap_sum += static_cast<double>(ev.mid_dist) - ev.bound_at_reject;
                ba.bound_sum += ev.bound_at_reject;
                if(ev.admitted_nn > 0) ++ba.successful_mids;
                else ++ba.failed_mids;
                if(ev.first_edge_type == HNSW_NGFix<T>::SearchEdgeType::HNSW_BASE) ++ba.first_h;
                else if(ev.first_edge_type == HNSW_NGFix<T>::SearchEdgeType::NGFIX) ++ba.first_n;
                else ++ba.first_r;

                const auto oit = mid_outcomes.find(ev.mid);
                if(oit != mid_outcomes.end()) {
                    const auto& mo = oit->second;
                    ba.direct_new_gt += mo.direct_new_gt;
                    ba.direct_shared_gt += mo.direct_shared_gt;
                    ba.recovered_desc_gt += mo.recovered_desc_gt_nearest;
                    if(mo.direct_new_gt > 0 || mo.recovered_desc_gt_nearest > 0) ++ba.useful_mids;
                }
            }

            for(size_t b = 0; b < BINS; ++b) {
                const auto& ba = bins[b];
                const double lo = static_cast<double>(b) / BINS;
                const double hi = static_cast<double>(b + 1) / BINS;
                *jump_hopbin_out
                    << query_id << "," << efs << "," << stats.cur_pop_count << ","
                    << result.paired_baseline_recall << "," << result.recall << ","
                    << b << "," << lo << "," << hi << ","
                    << ba.scans << "," << ba.successful_mids << "," << ba.failed_mids << ","
                    << ba.useful_mids << "," << ba.fresh_nn_ndc << "," << ba.admitted_nn << ","
                    << ba.direct_new_gt << "," << ba.direct_shared_gt << ","
                    << ba.recovered_desc_gt << ","
                    << ba.mid_gap_sum << "," << ba.bound_sum << ","
                    << ba.first_h << "," << ba.first_n << "," << ba.first_r << "\n";
            }
        }
    }

    const auto& ep = stats.twohop_examined_pairs;
    result.examined_pair_hh = ep.hnsw_hnsw;
    result.examined_pair_hn = ep.hnsw_ngfix;
    result.examined_pair_hr = ep.hnsw_rfix;
    result.examined_pair_nh = ep.ngfix_hnsw;
    result.examined_pair_nn = ep.ngfix_ngfix;
    result.examined_pair_nr = ep.ngfix_rfix;
    result.examined_pair_rh = ep.rfix_hnsw;
    result.examined_pair_rn = ep.rfix_ngfix;
    result.examined_pair_rr = ep.rfix_rfix;

    const auto& sp = stats.twohop_scored_pairs;
    result.scored_pair_hh = sp.hnsw_hnsw;
    result.scored_pair_hn = sp.hnsw_ngfix;
    result.scored_pair_hr = sp.hnsw_rfix;
    result.scored_pair_nh = sp.ngfix_hnsw;
    result.scored_pair_nn = sp.ngfix_ngfix;
    result.scored_pair_nr = sp.ngfix_rfix;
    result.scored_pair_rh = sp.rfix_hnsw;
    result.scored_pair_rn = sp.rfix_ngfix;
    result.scored_pair_rr = sp.rfix_rfix;

    const auto& jp = stats.jump_pairs;
    result.jump_pair_hh = jp.hnsw_hnsw;
    result.jump_pair_hn = jp.hnsw_ngfix;
    result.jump_pair_hr = jp.hnsw_rfix;
    result.jump_pair_nh = jp.ngfix_hnsw;
    result.jump_pair_nn = jp.ngfix_ngfix;
    result.jump_pair_nr = jp.ngfix_rfix;
    result.jump_pair_rh = jp.rfix_hnsw;
    result.jump_pair_rn = jp.rfix_ngfix;
    result.jump_pair_rr = jp.rfix_rfix;

    const auto& f = stats.first_hop_edges;
    const auto& s = stats.second_hop_edges;

    result.first_edges_total = f.examined_total;
    result.first_edges_hnsw = f.examined_hnsw_base;
    result.first_edges_ngfix = f.examined_ngfix;
    result.first_edges_rfix = f.examined_rfix;

    result.second_edges_total = s.examined_total;
    result.second_edges_hnsw = s.examined_hnsw_base;
    result.second_edges_ngfix = s.examined_ngfix;
    result.second_edges_rfix = s.examined_rfix;

    result.edges_total = result.first_edges_total + result.second_edges_total;
    result.edges_hnsw = result.first_edges_hnsw + result.second_edges_hnsw;
    result.edges_ngfix = result.first_edges_ngfix + result.second_edges_ngfix;
    result.edges_rfix = result.first_edges_rfix + result.second_edges_rfix;

    // Measurement-correctness stages. Baseline only has first-hop edges, but
    // summing first+second keeps the result schema well-defined for legacy modes.
    result.examined_total_m = f.examined_total + s.examined_total;
    result.examined_hnsw_m = f.examined_hnsw_base + s.examined_hnsw_base;
    result.examined_ngfix_m = f.examined_ngfix + s.examined_ngfix;
    result.examined_rfix_m = f.examined_rfix + s.examined_rfix;
    result.examined_ngfix_finite_m = f.examined_ngfix_finite_eh + s.examined_ngfix_finite_eh;
    result.examined_ngfix_inf_m = f.examined_ngfix_eh_inf + s.examined_ngfix_eh_inf;

    result.scored_total_m = f.scored_total + s.scored_total;
    result.scored_hnsw_m = f.scored_hnsw_base + s.scored_hnsw_base;
    result.scored_ngfix_m = f.scored_ngfix + s.scored_ngfix;
    result.scored_rfix_m = f.scored_rfix + s.scored_rfix;
    result.scored_ngfix_finite_m = f.scored_ngfix_finite_eh + s.scored_ngfix_finite_eh;
    result.scored_ngfix_inf_m = f.scored_ngfix_eh_inf + s.scored_ngfix_eh_inf;

    result.admitted_total_m = f.admitted_total + s.admitted_total;
    result.admitted_hnsw_m = f.admitted_hnsw_base + s.admitted_hnsw_base;
    result.admitted_ngfix_m = f.admitted_ngfix + s.admitted_ngfix;
    result.admitted_rfix_m = f.admitted_rfix + s.admitted_rfix;
    result.admitted_ngfix_finite_m = f.admitted_ngfix_finite_eh + s.admitted_ngfix_finite_eh;
    result.admitted_ngfix_inf_m = f.admitted_ngfix_eh_inf + s.admitted_ngfix_eh_inf;

    result.expanded_total_m = f.expanded_from_total + s.expanded_from_total;
    result.expanded_hnsw_m = f.expanded_from_hnsw_base + s.expanded_from_hnsw_base;
    result.expanded_ngfix_m = f.expanded_from_ngfix + s.expanded_from_ngfix;
    result.expanded_rfix_m = f.expanded_from_rfix + s.expanded_from_rfix;
    result.expanded_ngfix_finite_m = f.expanded_from_ngfix_finite_eh + s.expanded_from_ngfix_finite_eh;
    result.expanded_ngfix_inf_m = f.expanded_from_ngfix_eh_inf + s.expanded_from_ngfix_eh_inf;

    for(size_t eh = 0; eh < MAX_S + 2; ++eh) {
        result.examined_ngfix_eh_hist[eh] = f.examined_ngfix_eh_hist[eh] + s.examined_ngfix_eh_hist[eh];
        result.scored_ngfix_eh_hist[eh] = f.scored_ngfix_eh_hist[eh] + s.scored_ngfix_eh_hist[eh];
        result.admitted_ngfix_eh_hist[eh] = f.admitted_ngfix_eh_hist[eh] + s.admitted_ngfix_eh_hist[eh];
        result.expanded_ngfix_eh_hist[eh] = f.expanded_from_ngfix_eh_hist[eh] + s.expanded_from_ngfix_eh_hist[eh];
    }

    return result;
}


template<typename T>
void TestQueries(
    std::ostream& out,
    T* test_query,
    int* test_gt,
    size_t test_number,
    size_t k,
    size_t test_gt_d,
    size_t vecdim,
    HNSW_NGFix<T>* searcher,
    const std::string& mode,
    const std::vector<int>& requested_efs,
    const std::vector<float>& requested_dabs_gammas,
    typename HNSW_NGFix<T>::DABSDistanceTransform dabs_transform,
    std::ostream* eh_aggregate_out = nullptr,
    int event_trace_ef = -1,
    size_t event_trace_query_limit = std::numeric_limits<size_t>::max(),
    std::ostream* jump_event_out = nullptr,
    std::ostream* jump_mid_out = nullptr,
    std::ostream* jump_hopbin_out = nullptr,
    size_t query_offset = 0,
    bool event_trace_compact = false,
    std::ostream* nn_out = nullptr,
    std::ostream* search_state_out = nullptr)
{
    auto write_gate = [&](const BoundGateStats& g, double denominator) {
        out << "," << (searcher->bound_twohop_gate.active() || searcher->jump_policy.active())
            << "," << g.rejected_mids / denominator << "," << g.blocked_hop / denominator
            << "," << g.blocked_cur / denominator << "," << g.blocked_gap / denominator
            << "," << g.blocked_edge / denominator << "," << g.blocked_budget / denominator
            << "," << g.skipped_nn_edge / denominator
            << "," << g.blocked_policy / denominator << "," << g.tree_evaluations / denominator
            << "," << g.history_known / denominator << "," << g.history_unknown / denominator
            << "," << g.history_accepts / denominator
            << "," << g.blocked_detour / denominator
            << "," << g.blocked_selection / denominator
            << "," << g.blocked_trigger / denominator
            << "," << g.blocked_batch_budget / denominator
            << "," << g.eligible_mids / denominator
            << "," << g.selected_mids / denominator
            << "," << g.probed_base_mids / denominator
            << "," << g.probed_ngfix_mids / denominator
            << "," << g.probed_rfix_mids / denominator
            << "," << g.trigger_checks / denominator
            << "," << g.trigger_fires / denominator
            << "," << g.stall_fires / denominator
            << "," << g.frontier_fires / denominator
            << "," << g.onehop_topk_changes / denominator
            << "," << g.twohop_topk_changes / denominator
            << "," << g.nn_mid_cap_hits / denominator
            << "," << g.nn_cur_cap_hits / denominator
            << "," << g.nn_pool_targets / denominator << "," << g.nn_selected_targets / denominator
            << "," << g.nn_cheap_scores / denominator << "," << g.nn_cheap_products / denominator
            << "," << g.nn_shadow_scores / denominator << "," << g.nn_shadow_products / denominator
            << "," << g.nn_query_dims_inspected / denominator
            << "\n";
    };
    // One CSV contains both per-query rows and one summary row for each ef.
    // For QUERY rows qps = 1e6 / that query's latency_us.
    // For SUMMARY rows qps = Nqueries / total search time.
    out << "row_type,mode,ef,query_id,recall,ndc,ndc_entry,ndc_mid,ndc_nn,"
        << "ndc_nn_jump,ndc_nn_nojump,latency_us,qps,rderr,"
        << "hops,mid_examined,twohop_mid_scans,nn_checked,"
        << "direct_mids,jump_mids,failed_jump_mids,skipped_mids,"
        << "skipped_mid_from_success_probe,skipped_mid_from_failed_probe,skipped_mid_from_admitted_jump,jump_candidates,"
        << "jump_candidates_gt,final_gt_direct_jump,final_gt_bridge_descendant,final_gt_non_gt_bridge_root,final_topk_bridge_descendant,"
        << "paired_baseline_recall,paired_baseline_ndc,recovered_gt_total,recovered_gt_direct_jump,recovered_gt_bridge_descendant,recovered_gt_non_gt_bridge_root,lost_gt_total,"
        << "exam_pair_HH,exam_pair_HN,exam_pair_HR,exam_pair_NH,exam_pair_NN,exam_pair_NR,exam_pair_RH,exam_pair_RN,exam_pair_RR,"
        << "score_pair_HH,score_pair_HN,score_pair_HR,score_pair_NH,score_pair_NN,score_pair_NR,score_pair_RH,score_pair_RN,score_pair_RR,"
        << "jump_pair_HH,jump_pair_HN,jump_pair_HR,jump_pair_NH,jump_pair_NN,jump_pair_NR,jump_pair_RH,jump_pair_RN,jump_pair_RR,"
        << "edges_total,edges_hnsw,edges_ngfix,edges_rfix,"
        << "examined_total,examined_hnsw,examined_ngfix,examined_rfix,examined_ngfix_finite,examined_ngfix_inf,"
        << "scored_total,scored_hnsw,scored_ngfix,scored_rfix,scored_ngfix_finite,scored_ngfix_inf,"
        << "admitted_total,admitted_hnsw,admitted_ngfix,admitted_rfix,admitted_ngfix_finite,admitted_ngfix_inf,"
        << "expanded_total,expanded_hnsw,expanded_ngfix,expanded_rfix,expanded_ngfix_finite,expanded_ngfix_inf";
    for(const char* stage : {"exam", "score", "admit", "expand"}) {
        for(size_t b = 0; b < QUERY_EH_BUCKETS; ++b)
            out << "," << stage << "_" << QUERY_EH_BUCKET_NAMES[b];
    }
    out << ",dabs_gamma,jump_gate_active,gate_rejected_mids,gate_blocked_hop,gate_blocked_cur,"
        << "gate_blocked_gap,gate_blocked_edge,gate_blocked_budget,gate_skipped_nn_edge,"
        << "gate_blocked_policy,gate_tree_evaluations,gate_history_known,gate_history_unknown,gate_history_accepts,gate_blocked_detour,gate_blocked_selection,gate_blocked_trigger,gate_blocked_batch_budget,gate_eligible_mids,gate_selected_mids,gate_probed_base_mids,gate_probed_ngfix_mids,gate_probed_rfix_mids,gate_trigger_checks,gate_trigger_fires,gate_stall_fires,gate_frontier_fires,gate_onehop_topk_changes,gate_twohop_topk_changes,gate_nn_mid_cap_hits,gate_nn_cur_cap_hits,gate_nn_pool_targets,gate_nn_selected_targets,gate_nn_cheap_scores,gate_nn_cheap_products,gate_nn_shadow_scores,gate_nn_shadow_products,gate_nn_query_dims_inspected\n";

    if(eh_aggregate_out) {
        *eh_aggregate_out << "mode,ef,stage,eh,count,ratio_within_stage_ngfix,dabs_gamma\n";
    }

    if(jump_event_out) {
        *jump_event_out
            << "query_id,ef,total_hops,hop,hop_frac,cur_id,mid_id,nn_id,"
            << "cur_is_gt,mid_is_gt,nn_gt_rank,"
            << "cur_dist,mid_dist,nn_dist,bound_at_mid_reject,bound_before_nn_admit,"
            << "mid_gap,nn_margin,first_edge_type,second_edge_type,mid_checked_nn,mid_fresh_nn_cmps,"
            << "baseline_scored_nn,baseline_expanded_nn,nn_is_gt,baseline_final_gt,"
            << "bound_final,bound_final_gt,nearest_final_gt_descendants,nearest_recovered_gt_descendants,nearest_shared_gt_descendants,"
            << "ancestor_final_gt_descendants,ancestor_recovered_gt_descendants,ancestor_shared_gt_descendants,jump_class,"
            << "hop_over_ef,first_edge_eh_raw,second_edge_eh_raw,rank_in_fresh_nn_by_query_dist,"
            << "cur_total_degree,cur_hnsw_degree,cur_ngfix_degree,cur_rfix_degree,cur_repair_degree,cur_repair_ratio,cur_ngfix_finite_degree,cur_ngfix_inf_degree,"
            << "mid_total_degree,mid_hnsw_degree,mid_ngfix_degree,mid_rfix_degree,mid_repair_degree,mid_repair_ratio,mid_ngfix_finite_degree,mid_ngfix_inf_degree,"
            << "nn_total_degree,nn_hnsw_degree,nn_ngfix_degree,nn_rfix_degree,nn_repair_degree,nn_repair_ratio,nn_ngfix_finite_degree,nn_ngfix_inf_degree,"
            << "cur_mid_data_dist,mid_to_cur_edge_type,cur_mid_common_neighbors,cur_mid_union_neighbors,cur_mid_jaccard,"
            << "mid_nn_data_dist,nn_to_mid_edge_type,mid_nn_common_neighbors,mid_nn_union_neighbors,mid_nn_jaccard,"
            << "cur_nn_data_dist,cur_to_nn_edge_type,nn_to_cur_edge_type,cur_nn_common_neighbors,cur_nn_union_neighbors,cur_nn_jaccard\n";
    }
    if(search_state_out)
        *search_state_out<<std::setprecision(17)<<"query_id,ef,probe_id,cur_id,mid_id,hop,"
                         <<HNSW_NGFix<T>::boundSearchSnapshotHeader()<<'\n';
    if(nn_out) {
        *nn_out << std::setprecision(std::numeric_limits<float>::max_digits10);
        *nn_out << "query_id,ef,probe_id,hop,cur_id,mid_id,nn_id,adjacency_ordinal,first_type,second_type,second_eh,"
                << "pool_size,scored_size,censored,selected,scored,admitted,exact_dist,cheap_score,random_key,"
                << "exact_rank,direct_new_gt,nearest_recovered_desc_gt,value_label\n";
    }
    if(jump_mid_out && event_trace_compact) {
        *jump_mid_out << std::setprecision(std::numeric_limits<float>::max_digits10);
        *jump_mid_out << "query_id,ef,cur_id,mid_id,fresh_nn_cmps,useful_mid,direct_new_gt,"
                     << "nearest_recovered_desc_gt,admitted_nn,f0,f1,f2,f3,f4,f5,f6,f7\n";
    }
    if(jump_mid_out && !event_trace_compact) {
        *jump_mid_out
            << "query_id,ef,total_hops,hop,hop_frac,cur_id,mid_id,cur_dist,mid_dist,bound_at_reject,mid_gap,"
            << "first_edge_type,checked_nn,fresh_nn_cmps,admitted_nn,success,useful_mid,"
            << "baseline_scored_cur,baseline_expanded_cur,baseline_scored_mid,baseline_expanded_mid,"
            << "direct_new_gt,direct_shared_gt,sanity_gt_not_final,non_gt_roots,useful_roots_nearest,useful_roots_ancestor,"
            << "nearest_final_desc_gt,nearest_recovered_desc_gt,ancestor_final_desc_gt,ancestor_recovered_desc_gt,best_admitted_nn,best_admitted_nn_dist,"
            << "hop_over_ef,first_edge_eh_raw,"
            << "cur_total_degree,cur_hnsw_degree,cur_ngfix_degree,cur_rfix_degree,cur_repair_degree,cur_repair_ratio,cur_ngfix_finite_degree,cur_ngfix_inf_degree,"
            << "mid_total_degree,mid_hnsw_degree,mid_ngfix_degree,mid_rfix_degree,mid_repair_degree,mid_repair_ratio,mid_ngfix_finite_degree,mid_ngfix_inf_degree,"
            << "cur_mid_data_dist,mid_to_cur_edge_type,cur_mid_common_neighbors,cur_mid_union_neighbors,cur_mid_jaccard,"
            << "best_scored_nn,best_scored_nn_dist,best_scored_second_edge_type,best_scored_second_edge_eh_raw,best_scored_nn_is_gt,best_scored_baseline_scored,best_scored_baseline_final_gt,best_scored_bound_final_gt,"
            << "best_nn_total_degree,best_nn_hnsw_degree,best_nn_ngfix_degree,best_nn_rfix_degree,best_nn_repair_degree,best_nn_repair_ratio,best_nn_ngfix_finite_degree,best_nn_ngfix_inf_degree,"
            << "mid_best_nn_data_dist,best_nn_to_mid_edge_type,mid_best_nn_common_neighbors,mid_best_nn_union_neighbors,mid_best_nn_jaccard,"
            << "cur_best_nn_data_dist,cur_to_best_nn_edge_type,best_nn_to_cur_edge_type,cur_best_nn_common_neighbors,cur_best_nn_union_neighbors,cur_best_nn_jaccard\n";
    }
    if(jump_hopbin_out) {
        *jump_hopbin_out
            << "query_id,ef,total_hops,paired_baseline_recall,bound_recall,bin,hop_frac_lo,hop_frac_hi,"
            << "scans,successful_mids,failed_mids,useful_mids,fresh_nn_ndc,admitted_nn,"
            << "direct_new_gt,direct_shared_gt,recovered_desc_gt,mid_gap_sum,bound_sum,"
            << "first_h,first_n,first_r\n";
    }

    struct SearchConfig {
        int ef;
        float dabs_gamma;
    };
    std::vector<SearchConfig> configs;

    if(mode == "dabs") {
        if(requested_dabs_gammas.empty())
            throw std::runtime_error("DABS requires at least one --dabs_gammas value.");
        for(float gamma : requested_dabs_gammas)
            configs.push_back({0, gamma});
    } else if(event_trace_ef >= 0) {
        configs.push_back({event_trace_ef, std::numeric_limits<float>::quiet_NaN()});
    } else {
        const std::vector<int> default_efs = {100, 200, 300, 500, 800, 1000};
        const auto& efss = requested_efs.empty() ? default_efs : requested_efs;
        for(int ef : efss)
            configs.push_back({ef, std::numeric_limits<float>::quiet_NaN()});
    }
    out << std::fixed << std::setprecision(6);

    for(const auto& config : configs)
    {
        const int efs = config.ef;
        const float dabs_gamma = config.dabs_gamma;
        std::vector<SearchResult> results(test_number);

        for(size_t i = 0; i < test_number; i++)
        {
            auto gt = test_gt + i*test_gt_d;
            const bool trace_this_ef =
                (event_trace_ef >= 0 && efs == event_trace_ef && i < event_trace_query_limit);

            results[i] = TestSingleQuery(
                test_query + i*vecdim,
                gt,
                k,
                static_cast<size_t>(efs),
                dabs_gamma,
                dabs_transform,
                i + query_offset,
                searcher,
                mode,
                trace_this_ef ? jump_event_out : nullptr,
                trace_this_ef ? jump_mid_out : nullptr,
                trace_this_ef ? jump_hopbin_out : nullptr,
                event_trace_compact, trace_this_ef ? nn_out : nullptr,
                trace_this_ef ? search_state_out : nullptr);

            const auto& r = results[i];
            out << "QUERY," << mode << "," << efs << "," << i + query_offset << ","
                << r.recall << ","
                << r.ndc << ","
                << r.ndc_entry << ","
                << r.ndc_mid << ","
                << r.ndc_nn << ","
                << r.ndc_nn_jump << ","
                << r.ndc_nn_nojump << ","
                << r.latency_us << ","
                << r.qps << ","
                << r.rderr << ","
                << r.hops << ","
                << r.mid_examined << ","
                << r.twohop_mid_scans << ","
                << r.nn_checked << ","
                << r.direct_mids << ","
                << r.jump_mids << ","
                << r.failed_jump_mids << ","
                << r.skipped_mids << ","
                << r.skipped_mid_from_success_probe << ","
                << r.skipped_mid_from_failed_probe << ","
                << r.skipped_mid_from_admitted_jump << ","
                << r.jump_candidates << ","
                << r.jump_candidates_gt << ","
                << r.final_gt_direct_jump << ","
                << r.final_gt_bridge_descendant << ","
                << r.final_gt_non_gt_bridge_root << ","
                << r.final_topk_bridge_descendant << ","
                << r.paired_baseline_recall << ","
                << r.paired_baseline_ndc << ","
                << r.recovered_gt_total << ","
                << r.recovered_gt_direct_jump << ","
                << r.recovered_gt_bridge_descendant << ","
                << r.recovered_gt_non_gt_bridge_root << ","
                << r.lost_gt_total << ","
                << r.examined_pair_hh << "," << r.examined_pair_hn << "," << r.examined_pair_hr << ","
                << r.examined_pair_nh << "," << r.examined_pair_nn << "," << r.examined_pair_nr << ","
                << r.examined_pair_rh << "," << r.examined_pair_rn << "," << r.examined_pair_rr << ","
                << r.scored_pair_hh << "," << r.scored_pair_hn << "," << r.scored_pair_hr << ","
                << r.scored_pair_nh << "," << r.scored_pair_nn << "," << r.scored_pair_nr << ","
                << r.scored_pair_rh << "," << r.scored_pair_rn << "," << r.scored_pair_rr << ","
                << r.jump_pair_hh << "," << r.jump_pair_hn << "," << r.jump_pair_hr << ","
                << r.jump_pair_nh << "," << r.jump_pair_nn << "," << r.jump_pair_nr << ","
                << r.jump_pair_rh << "," << r.jump_pair_rn << "," << r.jump_pair_rr << ","
                << r.edges_total << ","
                << r.edges_hnsw << ","
                << r.edges_ngfix << ","
                << r.edges_rfix << ","
                << r.examined_total_m << "," << r.examined_hnsw_m << "," << r.examined_ngfix_m << "," << r.examined_rfix_m << "," << r.examined_ngfix_finite_m << "," << r.examined_ngfix_inf_m << ","
                << r.scored_total_m << "," << r.scored_hnsw_m << "," << r.scored_ngfix_m << "," << r.scored_rfix_m << "," << r.scored_ngfix_finite_m << "," << r.scored_ngfix_inf_m << ","
                << r.admitted_total_m << "," << r.admitted_hnsw_m << "," << r.admitted_ngfix_m << "," << r.admitted_rfix_m << "," << r.admitted_ngfix_finite_m << "," << r.admitted_ngfix_inf_m << ","
                << r.expanded_total_m << "," << r.expanded_hnsw_m << "," << r.expanded_ngfix_m << "," << r.expanded_rfix_m << "," << r.expanded_ngfix_finite_m << "," << r.expanded_ngfix_inf_m;

            for(const auto& hist : {r.examined_ngfix_eh_hist, r.scored_ngfix_eh_hist,
                                    r.admitted_ngfix_eh_hist, r.expanded_ngfix_eh_hist}) {
                const auto b = BucketEhHist(hist);
                for(size_t x : b) out << "," << x;
            }
            out << "," << dabs_gamma;
            write_gate(r.gate, 1.0);
        }

        double avg_recall = 0.0;
        double avg_ndc = 0.0;
        double avg_latency_ms = 0.0;
        double avg_rderr = 0.0;
        double aggregate_qps = 0.0;

        AllQueriesEvaluation(
            results,
            avg_recall,
            avg_ndc,
            avg_latency_ms,
            avg_rderr,
            aggregate_qps);

        // Compute averages for all detailed counters so the SUMMARY row has
        // the same schema as QUERY rows.
        double a_entry=0, a_mid=0, a_nn=0, a_nnj=0, a_nnnj=0;
        double a_hops=0, a_mide=0, a_tms=0, a_nnc=0;
        double a_dm=0, a_jm=0, a_fjm=0, a_sm=0, a_sms=0, a_smf=0, a_sma=0, a_jc=0;
        double a_jcgt=0, a_fgtdj=0, a_fgtbd=0, a_fgtnr=0, a_ftkbd=0;
        double a_pbr=0, a_pbndc=0, a_rgt=0, a_rgtd=0, a_rgtbd=0, a_rgtnr=0, a_lgt=0;
        double a_ephh=0, a_ephn=0, a_ephr=0, a_epnh=0, a_epnn=0, a_epnr=0, a_eprh=0, a_eprn=0, a_eprr=0;
        double a_sphh=0, a_sphn=0, a_sphr=0, a_spnh=0, a_spnn=0, a_spnr=0, a_sprh=0, a_sprn=0, a_sprr=0;
        double a_jphh=0, a_jphn=0, a_jphr=0, a_jpnh=0, a_jpnn=0, a_jpnr=0, a_jprh=0, a_jprn=0, a_jprr=0;
        double a_et=0, a_eh=0, a_en=0, a_er=0;
        double a_fet=0, a_feh=0, a_fen=0, a_fer=0;
        double a_set=0, a_seh=0, a_sen=0, a_ser=0;
        double total_latency_us=0;

        for(const auto& r : results) {
            a_entry += r.ndc_entry; a_mid += r.ndc_mid; a_nn += r.ndc_nn;
            a_nnj += r.ndc_nn_jump; a_nnnj += r.ndc_nn_nojump;
            a_hops += r.hops; a_mide += r.mid_examined;
            a_tms += r.twohop_mid_scans; a_nnc += r.nn_checked;
            a_dm += r.direct_mids; a_jm += r.jump_mids;
            a_fjm += r.failed_jump_mids; a_sm += r.skipped_mids;
            a_sms += r.skipped_mid_from_success_probe;
            a_smf += r.skipped_mid_from_failed_probe;
            a_sma += r.skipped_mid_from_admitted_jump;
            a_jc += r.jump_candidates;
            a_jcgt += r.jump_candidates_gt;
            a_fgtdj += r.final_gt_direct_jump;
            a_fgtbd += r.final_gt_bridge_descendant;
            a_fgtnr += r.final_gt_non_gt_bridge_root;
            a_ftkbd += r.final_topk_bridge_descendant;
            a_pbr += r.paired_baseline_recall;
            a_pbndc += r.paired_baseline_ndc;
            a_rgt += r.recovered_gt_total;
            a_rgtd += r.recovered_gt_direct_jump;
            a_rgtbd += r.recovered_gt_bridge_descendant;
            a_rgtnr += r.recovered_gt_non_gt_bridge_root;
            a_lgt += r.lost_gt_total;
            a_ephh += r.examined_pair_hh; a_ephn += r.examined_pair_hn; a_ephr += r.examined_pair_hr;
            a_epnh += r.examined_pair_nh; a_epnn += r.examined_pair_nn; a_epnr += r.examined_pair_nr;
            a_eprh += r.examined_pair_rh; a_eprn += r.examined_pair_rn; a_eprr += r.examined_pair_rr;
            a_sphh += r.scored_pair_hh; a_sphn += r.scored_pair_hn; a_sphr += r.scored_pair_hr;
            a_spnh += r.scored_pair_nh; a_spnn += r.scored_pair_nn; a_spnr += r.scored_pair_nr;
            a_sprh += r.scored_pair_rh; a_sprn += r.scored_pair_rn; a_sprr += r.scored_pair_rr;
            a_jphh += r.jump_pair_hh; a_jphn += r.jump_pair_hn; a_jphr += r.jump_pair_hr;
            a_jpnh += r.jump_pair_nh; a_jpnn += r.jump_pair_nn; a_jpnr += r.jump_pair_nr;
            a_jprh += r.jump_pair_rh; a_jprn += r.jump_pair_rn; a_jprr += r.jump_pair_rr;
            a_et += r.edges_total; a_eh += r.edges_hnsw;
            a_en += r.edges_ngfix; a_er += r.edges_rfix;
            a_fet += r.first_edges_total; a_feh += r.first_edges_hnsw;
            a_fen += r.first_edges_ngfix; a_fer += r.first_edges_rfix;
            a_set += r.second_edges_total; a_seh += r.second_edges_hnsw;
            a_sen += r.second_edges_ngfix; a_ser += r.second_edges_rfix;
            total_latency_us += r.latency_us;
        }

        const double denom = test_number > 0 ? static_cast<double>(test_number) : 1.0;
        const double avg_latency_us = total_latency_us / denom;

        // Aggregate measurement stages and exact EH histograms.
        SearchResult ms;
        for(const auto& r : results) {
            ms.gate += r.gate;
            ms.examined_total_m += r.examined_total_m; ms.examined_hnsw_m += r.examined_hnsw_m; ms.examined_ngfix_m += r.examined_ngfix_m; ms.examined_rfix_m += r.examined_rfix_m; ms.examined_ngfix_finite_m += r.examined_ngfix_finite_m; ms.examined_ngfix_inf_m += r.examined_ngfix_inf_m;
            ms.scored_total_m += r.scored_total_m; ms.scored_hnsw_m += r.scored_hnsw_m; ms.scored_ngfix_m += r.scored_ngfix_m; ms.scored_rfix_m += r.scored_rfix_m; ms.scored_ngfix_finite_m += r.scored_ngfix_finite_m; ms.scored_ngfix_inf_m += r.scored_ngfix_inf_m;
            ms.admitted_total_m += r.admitted_total_m; ms.admitted_hnsw_m += r.admitted_hnsw_m; ms.admitted_ngfix_m += r.admitted_ngfix_m; ms.admitted_rfix_m += r.admitted_rfix_m; ms.admitted_ngfix_finite_m += r.admitted_ngfix_finite_m; ms.admitted_ngfix_inf_m += r.admitted_ngfix_inf_m;
            ms.expanded_total_m += r.expanded_total_m; ms.expanded_hnsw_m += r.expanded_hnsw_m; ms.expanded_ngfix_m += r.expanded_ngfix_m; ms.expanded_rfix_m += r.expanded_rfix_m; ms.expanded_ngfix_finite_m += r.expanded_ngfix_finite_m; ms.expanded_ngfix_inf_m += r.expanded_ngfix_inf_m;
            for(size_t eh = 0; eh < MAX_S + 2; ++eh) {
                ms.examined_ngfix_eh_hist[eh] += r.examined_ngfix_eh_hist[eh];
                ms.scored_ngfix_eh_hist[eh] += r.scored_ngfix_eh_hist[eh];
                ms.admitted_ngfix_eh_hist[eh] += r.admitted_ngfix_eh_hist[eh];
                ms.expanded_ngfix_eh_hist[eh] += r.expanded_ngfix_eh_hist[eh];
            }
        }

        out << "SUMMARY," << mode << "," << efs << ",-1,"
            << avg_recall << ","
            << avg_ndc << ","
            << a_entry/denom << ","
            << a_mid/denom << ","
            << a_nn/denom << ","
            << a_nnj/denom << ","
            << a_nnnj/denom << ","
            << avg_latency_us << ","
            << aggregate_qps << ","
            << avg_rderr << ","
            << a_hops/denom << ","
            << a_mide/denom << ","
            << a_tms/denom << ","
            << a_nnc/denom << ","
            << a_dm/denom << ","
            << a_jm/denom << ","
            << a_fjm/denom << ","
            << a_sm/denom << ","
            << a_sms/denom << ","
            << a_smf/denom << ","
            << a_sma/denom << ","
            << a_jc/denom << ","
            << a_jcgt/denom << ","
            << a_fgtdj/denom << ","
            << a_fgtbd/denom << ","
            << a_fgtnr/denom << ","
            << a_ftkbd/denom << ","
            << a_pbr/denom << ","
            << a_pbndc/denom << ","
            << a_rgt/denom << ","
            << a_rgtd/denom << ","
            << a_rgtbd/denom << ","
            << a_rgtnr/denom << ","
            << a_lgt/denom << ","
            << a_ephh/denom << "," << a_ephn/denom << "," << a_ephr/denom << ","
            << a_epnh/denom << "," << a_epnn/denom << "," << a_epnr/denom << ","
            << a_eprh/denom << "," << a_eprn/denom << "," << a_eprr/denom << ","
            << a_sphh/denom << "," << a_sphn/denom << "," << a_sphr/denom << ","
            << a_spnh/denom << "," << a_spnn/denom << "," << a_spnr/denom << ","
            << a_sprh/denom << "," << a_sprn/denom << "," << a_sprr/denom << ","
            << a_jphh/denom << "," << a_jphn/denom << "," << a_jphr/denom << ","
            << a_jpnh/denom << "," << a_jpnn/denom << "," << a_jpnr/denom << ","
            << a_jprh/denom << "," << a_jprn/denom << "," << a_jprr/denom << ","
            << a_et/denom << ","
            << a_eh/denom << ","
            << a_en/denom << ","
            << a_er/denom << ","
            << ms.examined_total_m/denom << "," << ms.examined_hnsw_m/denom << "," << ms.examined_ngfix_m/denom << "," << ms.examined_rfix_m/denom << "," << ms.examined_ngfix_finite_m/denom << "," << ms.examined_ngfix_inf_m/denom << ","
            << ms.scored_total_m/denom << "," << ms.scored_hnsw_m/denom << "," << ms.scored_ngfix_m/denom << "," << ms.scored_rfix_m/denom << "," << ms.scored_ngfix_finite_m/denom << "," << ms.scored_ngfix_inf_m/denom << ","
            << ms.admitted_total_m/denom << "," << ms.admitted_hnsw_m/denom << "," << ms.admitted_ngfix_m/denom << "," << ms.admitted_rfix_m/denom << "," << ms.admitted_ngfix_finite_m/denom << "," << ms.admitted_ngfix_inf_m/denom << ","
            << ms.expanded_total_m/denom << "," << ms.expanded_hnsw_m/denom << "," << ms.expanded_ngfix_m/denom << "," << ms.expanded_rfix_m/denom << "," << ms.expanded_ngfix_finite_m/denom << "," << ms.expanded_ngfix_inf_m/denom;

        for(const auto& hist : {ms.examined_ngfix_eh_hist, ms.scored_ngfix_eh_hist,
                                ms.admitted_ngfix_eh_hist, ms.expanded_ngfix_eh_hist}) {
            const auto b = BucketEhHist(hist);
            for(size_t x : b) out << "," << (static_cast<double>(x) / denom);
        }
        out << "," << dabs_gamma;
        write_gate(ms.gate, denom);

        if(eh_aggregate_out) {
            const char* stage_names[4] = {"examined", "scored", "admitted", "expanded_from"};
            const std::array<size_t, MAX_S + 2>* hists[4] = {
                &ms.examined_ngfix_eh_hist, &ms.scored_ngfix_eh_hist,
                &ms.admitted_ngfix_eh_hist, &ms.expanded_ngfix_eh_hist
            };
            for(size_t si = 0; si < 4; ++si) {
                size_t stage_total = 0;
                for(size_t eh = 0; eh < MAX_S + 2; ++eh) stage_total += (*hists[si])[eh];
                const double stage_denom = stage_total > 0 ? static_cast<double>(stage_total) : 1.0;
                for(size_t eh = 0; eh <= MAX_S; ++eh) {
                    const size_t c = (*hists[si])[eh];
                    if(c) *eh_aggregate_out << mode << "," << efs << "," << stage_names[si] << "," << eh << "," << c << "," << (c/stage_denom) << "," << dabs_gamma << "\n";
                }
                const size_t c = (*hists[si])[MAX_S + 1];
                if(c) *eh_aggregate_out << mode << "," << efs << "," << stage_names[si] << ",INF," << c << "," << (c/stage_denom) << "," << dabs_gamma << "\n";
            }
        }

        std::cerr << "mode=" << mode
                  << " ef=" << efs
                  << " dabs_gamma=" << dabs_gamma
                  << " recall=" << avg_recall
                  << " ndc=" << avg_ndc
                  << " qps=" << aggregate_qps
                  << " avg_edges=" << a_et/denom
                  << " [HNSW=" << a_eh/denom
                  << ", NGFix=" << a_en/denom
                  << ", RFix=" << a_er/denom << "]\n";
    }
}
