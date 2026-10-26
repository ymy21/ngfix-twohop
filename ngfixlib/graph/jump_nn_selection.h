#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngfixlib {
enum class JumpNNSelection { FULL, PREFIX, RANDOM, CHEAP };
inline JumpNNSelection parseJumpNNSelection(const std::string& s) {
    if(s == "full") return JumpNNSelection::FULL;
    if(s == "prefix") return JumpNNSelection::PREFIX;
    if(s == "random") return JumpNNSelection::RANDOM;
    if(s == "cheap") return JumpNNSelection::CHEAP;
    throw std::invalid_argument("NN selection requires full, prefix, random or cheap.");
}
// Stable, seed-dependent permutation keys. No GT, exact query distances or
// training-query labels are used. Ties are broken by adjacency position.
inline uint64_t jumpNNHash(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
inline uint64_t jumpNNRandomKey(uint64_t seed, unsigned cur, unsigned mid, unsigned nn) {
    return jumpNNHash(jumpNNHash(seed ^ cur) ^ (uint64_t(mid) << 32) ^ nn);
}
struct JumpNNOption {
    unsigned id = 0;
    int neighbor_index = 0;
    float cheap_score = 0;
    uint64_t random_key = 0;
};
inline std::vector<int> selectJumpNNIndices(std::vector<JumpNNOption> pool,
                                           JumpNNSelection mode, size_t limit) {
    if(mode == JumpNNSelection::RANDOM)
        std::sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) {
            return a.random_key != b.random_key ? a.random_key < b.random_key
                                               : a.neighbor_index < b.neighbor_index;
        });
    else if(mode == JumpNNSelection::CHEAP)
        std::sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) {
            return a.cheap_score != b.cheap_score ? a.cheap_score < b.cheap_score
                                                 : a.neighbor_index < b.neighbor_index;
        });
    // FULL/PREFIX keep the physical adjacency order. Selected nodes are exact
    // scored in that order too: the ablation changes membership, not admission.
    if(mode != JumpNNSelection::FULL && pool.size() > limit) pool.resize(limit);
    std::vector<int> result;
    for(const auto& p : pool) result.push_back(p.neighbor_index);
    std::sort(result.begin(), result.end());
    return result;
}
class JumpCheapQuery {
    std::vector<size_t> coordinates;
    const float* query = nullptr;
    bool inner_product = true;
public:
    // Select once per query, using query values only. Coordinates are sorted
    // before vector access. This is a heuristic partial distance, not a bound.
    void prepare(const float* q, size_t dim, size_t count, bool ip) {
        if(!count || count > dim) throw std::invalid_argument("Cheap dimensions must be in [1,dim].");
        query = q; inner_product = ip;
        coordinates.resize(dim); std::iota(coordinates.begin(), coordinates.end(), 0);
        std::partial_sort(coordinates.begin(), coordinates.begin()+count, coordinates.end(),
            [&](size_t a, size_t b) {
                return std::abs(q[a]) != std::abs(q[b]) ? std::abs(q[a]) > std::abs(q[b]) : a < b;
            });
        coordinates.resize(count); std::sort(coordinates.begin(), coordinates.end());
    }
    float score(const float* x) const {
        float sum = 0;
        for(size_t i : coordinates) {
            if(inner_product) sum += query[i]*x[i];
            else { const float d=query[i]-x[i]; sum += d*d; }
        }
        return inner_product ? 1.0f-sum : sum;
    }
};
} // namespace ngfixlib
