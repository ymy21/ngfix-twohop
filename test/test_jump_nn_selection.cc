#include "ngfixlib/graph/jump_nn_selection.h"
#include <iostream>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool v) { if(!v) throw std::runtime_error("NN selection check failed"); }
int main() {
    std::vector<JumpNNOption> pool={{4,3,3,30},{5,6,0,20},{7,8,1,10}};
    check(selectJumpNNIndices(pool,JumpNNSelection::FULL,1)==std::vector<int>({3,6,8}));
    check(selectJumpNNIndices(pool,JumpNNSelection::PREFIX,2)==std::vector<int>({3,6}));
    check(selectJumpNNIndices(pool,JumpNNSelection::RANDOM,1)==std::vector<int>({8}));
    check(selectJumpNNIndices(pool,JumpNNSelection::CHEAP,2)==std::vector<int>({6,8}));
    check(selectJumpNNIndices({},JumpNNSelection::CHEAP,8).empty());
    pool[0].cheap_score=0;
    check(selectJumpNNIndices(pool,JumpNNSelection::CHEAP,1)==std::vector<int>({3}));
    float q[]={.1f,2,-3,0}, x[]={10,1,2,100}; JumpCheapQuery cheap;
    cheap.prepare(q,4,2,true); check(std::abs(cheap.score(x)-5)<1e-6);
    cheap.prepare(q,4,2,false); check(std::abs(cheap.score(x)-26)<1e-6);
    bool rejected=false; try { cheap.prepare(q,4,5,true); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    check(jumpNNRandomKey(17,3,4,5)==jumpNNRandomKey(17,3,4,5));
    check(jumpNNRandomKey(17,3,4,5)!=jumpNNRandomKey(29,3,4,5));
    std::cout << "NN selection membership, ties, reproducibility and partial scores passed.\n";
}
