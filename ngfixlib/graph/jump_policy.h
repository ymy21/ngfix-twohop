#pragma once

#include "bound_twohop_gate.h"
#include <array>
#include <fstream>
#include <vector>

namespace ngfixlib {

// V1 feature order is shared with analysis/train_jump_policy.py. Only values
// available BEFORE probing N(mid); no GT, future hops, or additional distances.
using JumpFeatures = std::array<float, 8>;
inline JumpFeatures makeJumpFeatures(float cur, float mid, float bound,
                                     size_t hop, size_t ef, unsigned type, uint16_t eh) {
    const double scale = BoundTwoHopGate::scale(bound);
    const bool finite_ngfix = type == 1 && eh != UINT16_MAX;
    return {static_cast<float>((static_cast<double>(mid) - bound) / scale),
            static_cast<float>((static_cast<double>(bound) - cur) / scale),
            static_cast<float>(static_cast<double>(hop ? hop - 1 : 0) / std::max(size_t(1), ef)),
            float(type == 0), float(type == 1), float(type == 2),
            finite_ngfix ? static_cast<float>(eh) : 0.0f,
            float(type == 1 && eh == UINT16_MAX)};
}

enum class JumpPolicyMode { OFF, TREE, HISTORY, HYBRID };
inline JumpPolicyMode parseJumpPolicy(const std::string& value) {
    if(value == "off") return JumpPolicyMode::OFF;
    if(value == "tree") return JumpPolicyMode::TREE;
    if(value == "history") return JumpPolicyMode::HISTORY;
    if(value == "hybrid") return JumpPolicyMode::HYBRID;
    throw std::invalid_argument("Invalid jump policy: " + value);
}

class JumpPolicy {
    struct TreeNode { int feature, left, right; double threshold; float score; };
    std::vector<TreeNode> tree_;
    // -1 means unseen/insufficient historical query support. Allocated only
    // for history/hybrid, once at load time. Four bytes per graph node.
    std::vector<float> history_;

    static void noExtraTokens(std::istream& in) {
        std::string extra;
        if(in >> extra) throw std::runtime_error("Trailing data in jump policy model");
    }
public:
    JumpPolicyMode mode = JumpPolicyMode::OFF;
    double tree_threshold = 0.0;
    double history_threshold = 0.0;
    size_t history_min_queries = 3;
    bool history_unknown_allow = true;

    bool active() const { return mode != JumpPolicyMode::OFF; }
    bool usesTree() const { return mode == JumpPolicyMode::TREE || mode == JumpPolicyMode::HYBRID; }
    bool usesHistory() const { return mode == JumpPolicyMode::HISTORY || mode == JumpPolicyMode::HYBRID; }

    void loadTree(const std::string& path, size_t k) {
        std::ifstream in(path);
        std::string magic;
        size_t features = 0, count = 0, model_k = 0;
        if(!(in >> magic >> features >> count >> model_k) || magic != "NGFIX_JUMP_TREE_V1" ||
           features != 8 || !count || count > 127 || model_k != k)
            throw std::runtime_error("Invalid jump tree header/K: " + path);
        tree_.resize(count);
        std::vector<unsigned> parents(count, 0), depths(count, 0);
        for(size_t i = 0; i < count; ++i) {
            auto& node = tree_[i];
            if(!(in >> node.feature >> node.left >> node.right >> node.threshold >> node.score) ||
               !BoundTwoHopGate::finite_value(node.threshold) ||
               !BoundTwoHopGate::finite_value(node.score) || node.score < 0 || node.score > 1)
                throw std::runtime_error("Invalid jump tree node");
            if(node.feature == -1) {
                if(node.left != -1 || node.right != -1) throw std::runtime_error("Invalid tree leaf");
            } else {
                if(node.feature < 0 || node.feature >= 8 || node.left <= static_cast<int>(i) ||
                   node.right <= static_cast<int>(i) || node.left >= static_cast<int>(count) ||
                   node.right >= static_cast<int>(count))
                    throw std::runtime_error("Invalid/cyclic jump tree");
                ++parents[node.left]; ++parents[node.right];
                depths[node.left] = depths[i] + 1;
                depths[node.right] = depths[i] + 1;
            }
            if((i && parents[i] != 1) || depths[i] > 6)
                throw std::runtime_error("Jump tree must be connected and depth <= 6");
        }
        noExtraTokens(in);
    }

    void loadHistory(const std::string& path, size_t nodes, size_t k) {
        std::ifstream in(path);
        std::string magic;
        size_t model_nodes = 0, model_k = 0, count = 0;
        if(!(in >> magic >> model_nodes >> model_k >> count) || magic != "NGFIX_JUMP_HISTORY_V1" ||
           model_nodes != nodes || model_k != k || count > nodes)
            throw std::runtime_error("Invalid jump history header/node count/K: " + path);
        history_.assign(nodes, -1.0f);
        size_t previous = 0;
        for(size_t i = 0; i < count; ++i) {
            size_t id = 0, support = 0;
            float score = 0;
            if(!(in >> id >> score >> support) || id >= nodes || (i && id <= previous) ||
               !BoundTwoHopGate::finite_value(score) || score < 0 || score > 1)
                throw std::runtime_error("Invalid jump history row");
            previous = id;
            if(support >= history_min_queries) history_[id] = score;
        }
        noExtraTokens(in);
    }

    void validate() const {
        if(!BoundTwoHopGate::finite_value(tree_threshold) || tree_threshold < 0 ||
           !BoundTwoHopGate::finite_value(history_threshold) || history_threshold < 0 ||
           (usesTree() && tree_.empty()) || (usesHistory() && history_.empty()))
            throw std::runtime_error("Missing/invalid jump policy model or threshold");
    }

    // A history hit can rescue a tree rejection in HYBRID. Unknown history
    // always falls back to the tree in HYBRID, regardless of unknown_allow.
    bool allows(size_t cur, float cur_dist, float mid_dist, float bound, size_t hop,
                size_t ef, unsigned edge_type, uint16_t eh, BoundGateStats& stats) const {
        bool known = false, history_pass = false;
        if(usesHistory()) {
            known = cur < history_.size() && history_[cur] >= 0;
            if(known) ++stats.history_known;
            else ++stats.history_unknown;
            history_pass = known && history_[cur] >= history_threshold;
            if(mode == JumpPolicyMode::HISTORY)
                return known ? history_pass : history_unknown_allow;
            if(history_pass) { ++stats.history_accepts; return true; }
        }
        if(usesTree()) {
            ++stats.tree_evaluations;
            const auto features = makeJumpFeatures(cur_dist, mid_dist, bound, hop, ef, edge_type, eh);
            size_t index = 0;
            while(tree_[index].feature >= 0) {
                const auto& node = tree_[index];
                index = features[node.feature] <= node.threshold ? node.left : node.right;
            }
            return tree_[index].score >= tree_threshold;
        }
        return true;
    }
};

} // namespace ngfixlib
