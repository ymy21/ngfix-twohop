#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include <deque>

#include "jump_nn_selection.h"

namespace ngfixlib {

// NN selection is orthogonal to mid selection and trigger.
enum class JumpSchedule { IMMEDIATE, AFTER_ONEHOP };
enum class JumpMidSelection { ADJACENCY, GAP, STRATIFIED };
enum class JumpTrigger { ALWAYS, STALL, STALL_FRONTIER };
inline JumpSchedule parseJumpSchedule(const std::string& s) {
    if(s == "immediate") return JumpSchedule::IMMEDIATE;
    if(s == "after_onehop") return JumpSchedule::AFTER_ONEHOP;
    throw std::invalid_argument("Invalid jump schedule: " + s);
}
inline JumpMidSelection parseJumpMidSelection(const std::string& s) {
    if(s == "adjacency") return JumpMidSelection::ADJACENCY;
    if(s == "gap") return JumpMidSelection::GAP;
    if(s == "stratified") return JumpMidSelection::STRATIFIED;
    throw std::invalid_argument("Invalid mid selection: " + s);
}
inline JumpTrigger parseJumpTrigger(const std::string& s) {
    if(s == "always") return JumpTrigger::ALWAYS;
    if(s == "stall") return JumpTrigger::STALL;
    if(s == "stall_frontier") return JumpTrigger::STALL_FRONTIER;
    throw std::invalid_argument("Invalid jump trigger: " + s);
}

struct PendingJumpMid { unsigned id; float distance; int neighbor_index; unsigned type; };

// At selection time all mids share one live bound, so sorting distances is
// equivalent to sorting gap ratios. Tie-break by original adjacency ordinal.
inline std::vector<size_t> selectJumpMids(const std::vector<PendingJumpMid>& mids,
                                        JumpMidSelection mode, size_t limit) {
    std::vector<size_t> order;
    for(size_t i = 0; i < mids.size(); ++i) order.push_back(i);
    const size_t count = limit ? std::min(limit, mids.size()) : mids.size();
    if(mode == JumpMidSelection::ADJACENCY) { order.resize(count); return order; }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return mids[a].distance != mids[b].distance ? mids[a].distance < mids[b].distance
                                                   : mids[a].neighbor_index < mids[b].neighbor_index;
    });
    if(mode == JumpMidSelection::GAP) { order.resize(count); return order; }
    std::vector<size_t> result;
    bool base = false, repair = false;
    for(size_t i : order) {
        bool& seen = mids[i].type == 0 ? base : repair; // NGFix and RFix share repair quota.
        if(!seen && result.size() < count) { result.push_back(i); seen = true; }
    }
    for(size_t i : order) {
        if(result.size() == count) break;
        if(std::find(result.begin(), result.end(), i) == result.end()) result.push_back(i);
    }
    std::sort(result.begin(), result.end(), [&](size_t a, size_t b) {
        return mids[a].distance != mids[b].distance ? mids[a].distance < mids[b].distance
                                                   : mids[a].neighbor_index < mids[b].neighbor_index;
    });
    return result;
}

class JumpProgressWindow {
    size_t width_, sum_ = 0;
    std::deque<size_t> changes_;
public:
    explicit JumpProgressWindow(size_t width) : width_(width) {}
    void observe(size_t onehop_changes, bool topk_full) {
        if(!topk_full) { changes_.clear(); sum_ = 0; return; }
        changes_.push_back(onehop_changes); sum_ += onehop_changes;
        if(changes_.size() > width_) { sum_ -= changes_.front(); changes_.pop_front(); }
    }
    bool stalled() const { return width_ && changes_.size() == width_ && sum_ == 0; }
};

// These filters describe stored construction provenance, not current-query EH.
enum class JumpEdgeFilter { ALL, BASE, NGFIX, NGFIX_FINITE, NGFIX_INF, RFIX };

inline JumpEdgeFilter parseJumpEdgeFilter(const std::string& value) {
    if(value == "all") return JumpEdgeFilter::ALL;
    if(value == "base") return JumpEdgeFilter::BASE;
    if(value == "ngfix") return JumpEdgeFilter::NGFIX;
    if(value == "ngfix_finite") return JumpEdgeFilter::NGFIX_FINITE;
    if(value == "ngfix_inf") return JumpEdgeFilter::NGFIX_INF;
    if(value == "rfix") return JumpEdgeFilter::RFIX;
    throw std::invalid_argument("Invalid jump edge filter: " + value);
}

inline bool jumpEdgeAllowed(JumpEdgeFilter filter, unsigned type, uint16_t eh) {
    // SearchEdgeType: BASE=0, NGFIX=1, RFIX=2. Classify RFix before EH_INF.
    switch(filter) {
        case JumpEdgeFilter::ALL: return true;
        case JumpEdgeFilter::BASE: return type == 0;
        case JumpEdgeFilter::NGFIX: return type == 1;
        case JumpEdgeFilter::NGFIX_FINITE:
            return type == 1 && eh != std::numeric_limits<uint16_t>::max();
        case JumpEdgeFilter::NGFIX_INF:
            return type == 1 && eh == std::numeric_limits<uint16_t>::max();
        case JumpEdgeFilter::RFIX: return type == 2;
    }
    return false;
}

struct BoundTwoHopGate {
    // All gates are opt-in: defaults reproduce the original boundtwohop.
    size_t cur_topk = 0;             // 0 disables the observed-distance rank gate
    size_t min_hops = 0;             // first permitted expanded-cur ordinal
    double cur_margin = 0.0;         // minimum (beam_bound - cur_dist) / scale
    // Finite sentinel: this repository builds with -Ofast, which assumes finite
    // floating-point values. Do not use isfinite(infinity) as an enable switch.
    double mid_gap = std::numeric_limits<double>::max();
    // Require positive cur slack and (mid-bound) <= ratio * (bound-cur).
    // Unlike mid_gap this is invariant to positive affine distance transforms.
    // It is a heuristic, not a metric bound or a recall guarantee.
    double detour_ratio = std::numeric_limits<double>::max();
    size_t max_mids_per_cur = 0;     // 0 unlimited; immediate uses adjacency, deferred uses selection
    size_t max_nn_per_query = 0;     // exact second-hop distance budget; 0 = unlimited
    JumpEdgeFilter first_edge = JumpEdgeFilter::ALL;
    JumpEdgeFilter second_edge = JumpEdgeFilter::ALL;
    JumpSchedule schedule = JumpSchedule::IMMEDIATE;
    JumpMidSelection selection = JumpMidSelection::ADJACENCY;
    JumpTrigger trigger = JumpTrigger::ALWAYS;
    size_t stall_window = 8;
    size_t max_nn_per_mid = 0;      // Fresh full distances, not neighbor IDs examined.
    size_t max_nn_per_cur = 0;      // Optional batch cap shared by this cur's probes.

    JumpNNSelection nn_selection = JumpNNSelection::FULL;
    size_t nn_limit = 8;
    size_t nn_cheap_dims = 16;
    uint64_t nn_seed = 17;

    bool active() const {
        return cur_topk || min_hops || cur_margin > 0 || has_mid_gap() || has_detour_ratio() ||
               max_mids_per_cur || max_nn_per_query ||
               first_edge != JumpEdgeFilter::ALL || second_edge != JumpEdgeFilter::ALL ||
               schedule != JumpSchedule::IMMEDIATE || max_nn_per_mid || max_nn_per_cur ||
               nn_selection != JumpNNSelection::FULL;
    }

    bool has_mid_gap() const { return mid_gap < std::numeric_limits<double>::max(); }
    bool has_detour_ratio() const { return detour_ratio < std::numeric_limits<double>::max(); }

    bool allows_detour(float cur, float mid, float bound) const {
        if(!has_detour_ratio()) return true;
        const double slack = static_cast<double>(bound) - cur;
        return slack > 0 && static_cast<double>(mid) - bound <= detour_ratio * slack;
    }

    static bool finite_value(double value) {
        return (std::bit_cast<uint64_t>(value) & UINT64_C(0x7ff0000000000000)) !=
               UINT64_C(0x7ff0000000000000);
    }

    void validate() const {
        if(!finite_value(cur_margin) || cur_margin < 0 ||
           !finite_value(mid_gap) || mid_gap < 0 ||
           !finite_value(detour_ratio) || detour_ratio < 0)
            throw std::invalid_argument("Jump margins must be finite and non-negative.");
        if(schedule == JumpSchedule::IMMEDIATE &&
           (selection != JumpMidSelection::ADJACENCY || trigger != JumpTrigger::ALWAYS))
            throw std::invalid_argument("Selection/trigger require --jump_schedule after_onehop.");
        if(!nn_limit || !nn_cheap_dims) throw std::invalid_argument("NN limit and cheap dimensions must be positive.");
        if(!stall_window) throw std::invalid_argument("Jump stall window must be positive.");
        if(selection == JumpMidSelection::STRATIFIED && max_mids_per_cur < 2)
            throw std::invalid_argument("Stratified selection requires a mid cap >= 2.");
    }

    static double scale(float bound) {
        // Sign-safe also for non-unit IP. This remains a heuristic, not a
        // metric-distance bound or triangle-inequality guarantee.
        return std::max(std::abs(static_cast<double>(bound)), 1e-6);
    }
};

// Track the r smallest distances among ALL freshly scored live nodes, including
// beam-rejected ones. No extra vector distances, GT labels, or final-state data.
class ObservedDistanceTopK {
    size_t capacity_;
    std::priority_queue<float> best_;
public:
    explicit ObservedDistanceTopK(size_t capacity) : capacity_(capacity) {}
    bool observe(float distance, bool deleted = false) {
        if(!capacity_ || deleted) return false;
        if(best_.size() < capacity_) { best_.push(distance); return true; }
        if(distance < best_.top()) { best_.pop(); best_.push(distance); return true; }
        return false;
    }
    bool full() const { return capacity_ && best_.size() == capacity_; }
    bool allows(float distance) const {
        return !capacity_ || (best_.size() == capacity_ && distance <= best_.top());
    }
};

struct BoundGateStats {
    size_t rejected_mids = 0;
    size_t blocked_hop = 0;
    size_t blocked_cur = 0;
    size_t blocked_gap = 0;
    size_t blocked_detour = 0;
    size_t blocked_edge = 0;
    size_t blocked_budget = 0;
    size_t skipped_nn_edge = 0;
    size_t blocked_policy = 0;
    size_t tree_evaluations = 0;
    size_t history_known = 0;
    size_t history_unknown = 0;
    size_t history_accepts = 0; // hybrid passed on history without consulting tree
    size_t blocked_selection = 0, blocked_trigger = 0, blocked_batch_budget = 0;
    size_t eligible_mids = 0, selected_mids = 0;
    size_t probed_base_mids = 0, probed_ngfix_mids = 0, probed_rfix_mids = 0;
    size_t trigger_checks = 0, trigger_fires = 0, stall_fires = 0, frontier_fires = 0;
    size_t onehop_topk_changes = 0, twohop_topk_changes = 0;
    size_t nn_mid_cap_hits = 0, nn_cur_cap_hits = 0;

    size_t nn_pool_targets = 0, nn_selected_targets = 0;
    size_t nn_cheap_scores = 0, nn_cheap_products = 0;
    size_t nn_shadow_scores = 0, nn_shadow_products = 0;
    size_t nn_query_dims_inspected = 0;

    BoundGateStats& operator+=(const BoundGateStats& other) {
        rejected_mids += other.rejected_mids;
        blocked_hop += other.blocked_hop;
        blocked_cur += other.blocked_cur;
        blocked_gap += other.blocked_gap;
        blocked_detour += other.blocked_detour;
        blocked_edge += other.blocked_edge;
        blocked_budget += other.blocked_budget;
        skipped_nn_edge += other.skipped_nn_edge;
        blocked_policy += other.blocked_policy;
        tree_evaluations += other.tree_evaluations;
        history_known += other.history_known;
        history_unknown += other.history_unknown;
        history_accepts += other.history_accepts;
        blocked_selection += other.blocked_selection;
        blocked_trigger += other.blocked_trigger;
        blocked_batch_budget += other.blocked_batch_budget;
        eligible_mids += other.eligible_mids; selected_mids += other.selected_mids;
        probed_base_mids += other.probed_base_mids;
        probed_ngfix_mids += other.probed_ngfix_mids;
        probed_rfix_mids += other.probed_rfix_mids;
        trigger_checks += other.trigger_checks; trigger_fires += other.trigger_fires;
        stall_fires += other.stall_fires; frontier_fires += other.frontier_fires;
        onehop_topk_changes += other.onehop_topk_changes;
        twohop_topk_changes += other.twohop_topk_changes;
        nn_mid_cap_hits += other.nn_mid_cap_hits; nn_cur_cap_hits += other.nn_cur_cap_hits;
        nn_pool_targets += other.nn_pool_targets; nn_selected_targets += other.nn_selected_targets;
        nn_cheap_scores += other.nn_cheap_scores; nn_cheap_products += other.nn_cheap_products;
        nn_shadow_scores += other.nn_shadow_scores; nn_shadow_products += other.nn_shadow_products;
        nn_query_dims_inspected += other.nn_query_dims_inspected;
        return *this;
    }
};

} // namespace ngfixlib
