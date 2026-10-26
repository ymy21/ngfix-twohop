#include "ngfixlib/graph/bound_twohop_gate.h"
#include <iostream>
#include <stdexcept>
using namespace ngfixlib;
static void check(bool value) { if(!value) throw std::runtime_error("jump schedule check failed"); }
int main() {
    std::vector<PendingJumpMid> pool = {{9,.4f,1,1},{8,.3f,2,2},{7,.5f,3,0},{6,.3f,4,0}};
    check(selectJumpMids(pool,JumpMidSelection::ADJACENCY,2)==std::vector<size_t>({0,1}));
    check(selectJumpMids(pool,JumpMidSelection::GAP,2)==std::vector<size_t>({1,3}));
    check(selectJumpMids(pool,JumpMidSelection::STRATIFIED,2)==std::vector<size_t>({1,3}));
    pool[3].distance=.6f;
    check(selectJumpMids(pool,JumpMidSelection::STRATIFIED,2)==std::vector<size_t>({1,2}));
    pool[2].type=1; pool[3].type=2; // Missing base quota must backfill, without duplicates.
    check(selectJumpMids(pool,JumpMidSelection::STRATIFIED,2)==std::vector<size_t>({1,0}));
    check(selectJumpMids({},JumpMidSelection::GAP,2).empty());
    check(selectJumpMids(pool,JumpMidSelection::GAP,0).size()==4);
    JumpProgressWindow window(2);
    window.observe(0,true); check(!window.stalled());
    window.observe(1,true); check(!window.stalled());
    window.observe(0,true); check(!window.stalled());
    window.observe(0,true); check(window.stalled());
    window.observe(0,false); check(!window.stalled());
    ObservedDistanceTopK top(2);
    check(top.observe(3)); check(!top.full());
    check(!top.observe(1,true)); check(!top.full());
    check(top.observe(2)); check(top.full());
    check(!top.observe(4)); check(!top.observe(3));
    check(top.observe(1)); check(top.allows(2)); check(!top.allows(3));
    BoundTwoHopGate gate; gate.validate(); check(!gate.active());
    gate.selection=JumpMidSelection::GAP;
    bool rejected=false; try { gate.validate(); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    gate.schedule=JumpSchedule::AFTER_ONEHOP; gate.validate(); check(gate.active());
    gate.selection=JumpMidSelection::STRATIFIED;
    rejected=false; try { gate.validate(); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected); gate.max_mids_per_cur=2; gate.validate();
    std::cout << "Selection, quota backfill, progress warmup, top-K changes and validation passed.\n";
}
