// Included inside HNSW_NGFix<T>. Only cur/mid selection; all fresh NN scored.
static BoundTwoHopGate curMidFullProfile(double gap, double margin,
        double ratio=std::numeric_limits<double>::max(), size_t mid_cap=0) {
    BoundTwoHopGate g;
    g.schedule=JumpSchedule::AFTER_ONEHOP;
    g.selection=JumpMidSelection::GAP;
    g.trigger=JumpTrigger::ALWAYS;
    g.mid_gap=gap; g.cur_margin=margin; g.detour_ratio=ratio;
    g.max_mids_per_cur=mid_cap;
    g.nn_selection=JumpNNSelection::FULL;
    g.validate(); return g;
}

// Configure with curMidFullProfile(). Count=false removes measurement counters.
// There are no query, per-cur or per-mid NN scoring budgets.
// Equivalent results/NDC to searchKnnBoundTwoHop with this profile, without
// attribution maps, shadow distances, feature collection, or GT lookups.
struct CurMidFullCounters {
    size_t entry=0, mid=0, nn=0, rejected=0, blocked_cur=0, blocked_gap=0,
        blocked_ratio=0, blocked_selection=0, probes=0, successful_probes=0,
        admitted_nn=0, scanned_nn=0, expanded=0, blocked_mark=0,
        mark_seen=0, mark_unknown=0, mark_allowed=0, nn_zero_admission=0;
};
template<bool CountDistances=false, bool UseAB=false>
std::vector<std::pair<float,id_t>> searchKnnCurMidFullClean(
        T* query_data, size_t k, size_t ef, size_t& ndc,
        CurMidFullCounters* counters=nullptr, const ABMarks* marks=nullptr,
        ABMode mark_mode=ABMode::NONE) {
    if constexpr(UseAB) if(!marks || mark_mode==ABMode::NONE)
        throw std::runtime_error("AB specialization requires marks and A/B mode");
    const auto& g=bound_twohop_gate;
    size_t hop_limit=std::numeric_limits<int>::max(),hop=0;
    if(ef<k) { hop_limit=ef; ef=k; }
    Search_QuadHeap q0(ef,visited_list_pool_); auto q=&q0;
    const float entry_dist=getQueryDist(entry_point,query_data);
    if constexpr(CountDistances) { ++ndc; if(counters) ++counters->entry; }
    q->push(entry_point,entry_dist,is_deleted(entry_point)); q->set_visited(entry_point);
    while(!q->is_empty()) {
        if(hop>hop_limit) break;
        const auto current=q->get_next_id();
        const id_t cur=current.second; const float cur_dist=-current.first;
        if(cur_dist>q->get_dist_bound()) break;
        ++hop;
        if constexpr(CountDistances) if(counters) ++counters->expanded;
        std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
        auto [outs,sz,st]=getNeighbors(cur);
        std::vector<PendingJumpMid> pending; pending.reserve(sz);
        // Complete ordinary one-hop before touching ANY rejected mid's list.
        for(int i=st;i<st+sz;++i) {
            const id_t mid=outs[i];
            if(i<st+sz-1) {
                q->prefetch_visited_list(outs[i+1]);
                __builtin_prefetch(getData(outs[i+1]),0,1);
            }
            if(q->is_visited(mid)) continue;
            q->set_visited(mid);
            const float distance=getQueryDist(mid,query_data);
            if constexpr(CountDistances) { ++ndc; if(counters) ++counters->mid; }
            if(q->top_candidates.size()<q->L || q->get_dist_bound()>distance)
                q->push(mid,distance,is_deleted(mid));
            else {
                pending.push_back({mid,distance,i,0});
                if constexpr(CountDistances) if(counters) ++counters->rejected;
            }
        }
        auto allows=[&](float mid_dist) {
            const float bound=q->get_dist_bound();
            const double scale=BoundTwoHopGate::scale(bound);
            if(g.cur_margin>0 && double(bound)-cur_dist<g.cur_margin*scale) {
                if constexpr(CountDistances) if(counters) ++counters->blocked_cur;
                return false;
            }
            if(g.has_mid_gap() && double(mid_dist)-bound>g.mid_gap*scale) {
                if constexpr(CountDistances) if(counters) ++counters->blocked_gap;
                return false;
            }
            if(!g.allows_detour(cur_dist,mid_dist,bound)) {
                if constexpr(CountDistances) if(counters) ++counters->blocked_ratio;
                return false;
            }
            return true;
        };
        std::vector<PendingJumpMid> eligible;
        for(const auto& m:pending) {
            if constexpr(UseAB) {
                const uint8_t flag=marks->at(cur,size_t(m.neighbor_index-st));
                const bool allowed=ABMarks::allows(flag,mark_mode);
                if constexpr(CountDistances) if(counters) {
                    if(flag&1) ++counters->mark_seen; else ++counters->mark_unknown;
                    if(allowed) ++counters->mark_allowed; else ++counters->blocked_mark;
                }
                if(!allowed) continue;
            }
            if(allows(m.distance)) eligible.push_back(m);
        }
        const auto chosen=selectJumpMids(eligible,JumpMidSelection::GAP,g.max_mids_per_cur);
        if constexpr(CountDistances) if(counters) counters->blocked_selection+=eligible.size()-chosen.size();
        for(size_t index:chosen) {
            const auto& mid=eligible[index];
            // Probe admission tightens bound: recheck cur AND mid gates live.
            if(!allows(mid.distance)) continue;
            if constexpr(CountDistances) if(counters) ++counters->probes;
            bool successful=false;
            const size_t nn_before=(CountDistances && counters)?counters->nn:0;
            struct Candidate { id_t id; float dist; };
            std::vector<Candidate> candidates;
            std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid.id]);
            auto [nn_outs,nn_sz,nn_st]=getNeighbors(mid.id);
            // Score every fresh neighbor of this selected mid; visited IDs are skipped.
            // There is no prefix, sampling, cheap-distance filter or NN cap.
            for(int j=nn_st;j<nn_st+nn_sz;++j) {
                if constexpr(CountDistances) if(counters) ++counters->scanned_nn;
                const id_t nn=nn_outs[j];
                if(q->is_visited(nn)) continue;
                if(j<nn_st+nn_sz-1) {
                    q->prefetch_visited_list(nn_outs[j+1]);
                    __builtin_prefetch(getData(nn_outs[j+1]),0,1);
                }
                q->set_visited(nn);
                const float distance=getQueryDist(nn,query_data);
                if constexpr(CountDistances) { ++ndc; if(counters) ++counters->nn; }
                candidates.push_back({nn,distance});
            }
            std::sort(candidates.begin(),candidates.end(),
                [](const Candidate& a,const Candidate& b) { return a.dist<b.dist; });
            for(const auto& c:candidates)
                if(q->top_candidates.size()<q->L || q->get_dist_bound()>c.dist) {
                    q->push(c.id,c.dist,is_deleted(c.id));
                    successful=true;
                    if constexpr(CountDistances) if(counters) ++counters->admitted_nn;
                }
            if constexpr(CountDistances) if(counters && successful) ++counters->successful_probes;
            if constexpr(CountDistances) if(counters && !successful)
                counters->nn_zero_admission+=counters->nn-nn_before;
        }
    }
    q->releaseVisitedList(); return q->get_result(k);
}
