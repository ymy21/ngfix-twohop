// Included inside HNSW_NGFix<T>. Existing diagnostic search methods are unchanged.
// These kernels support the frozen profile only; no trace/attribution objects.
static BoundTwoHopGate clean32Profile() {
    BoundTwoHopGate g;
    g.schedule=JumpSchedule::AFTER_ONEHOP;
    g.selection=JumpMidSelection::GAP;
    g.trigger=JumpTrigger::ALWAYS;
    g.mid_gap=0.06;
    g.max_mids_per_cur=2;
    g.max_nn_per_query=4000;
    g.nn_selection=JumpNNSelection::CHEAP;
    g.nn_limit=8;
    g.nn_cheap_dims=32;
    return g;
}

// Same heap/visited/neighbor order/prefetch/push/stop rules as searchKnn().
// Counting is compiled out of the timed instantiation. The counted version
// includes the entry distance, matching searchKnnWithStats's accounting.
template<bool CountDistances=false>
std::vector<std::pair<float,id_t>> searchKnnBeamClean(
        T* query_data,size_t k,size_t ef,size_t& ndc) {
    size_t hop_limit=std::numeric_limits<int>::max(),hop=0;
    if(ef<k) { hop_limit=ef; ef=k; }
    Search_QuadHeap q0(ef,visited_list_pool_); auto q=&q0;
    const float entry_dist=getQueryDist(entry_point,query_data);
    if constexpr(CountDistances) ++ndc;
    q->push(entry_point,entry_dist,is_deleted(entry_point));
    q->set_visited(entry_point);
    while(!q->is_empty()) {
        if(hop>hop_limit) break;
        const auto current=q->get_next_id();
        const id_t cur=current.second;
        if(-current.first>q->get_dist_bound()) break;
        ++hop;
        std::shared_lock<std::shared_mutex> lock(node_locks[cur]);
        auto [outs,sz,st]=getNeighbors(cur);
        for(int i=st;i<st+sz;++i) {
            const id_t candidate=outs[i];
            if(i<st+sz-1) q->prefetch_visited_list(outs[i+1]);
            if(!q->is_visited(candidate)) {
                q->set_visited(candidate);
                const float dist=getQueryDist(candidate,query_data);
                q->push(candidate,dist,is_deleted(candidate));
                if constexpr(CountDistances) ++ndc;
            }
        }
    }
    q->releaseVisitedList();
    return q->get_result(k);
}

template<bool CountDistances=false>
std::vector<std::pair<float,id_t>> searchKnnCheap32Clean(
        T* query_data,size_t k,size_t ef,size_t& ndc) {
    static_assert(std::is_same_v<T,float>,"Clean cheap32 requires raw float vectors.");
    const auto gate=clean32Profile();
    JumpCheapQuery cheap_query;
    const bool ip=dynamic_cast<IPSpace_float*>(query_space)!=nullptr;
    cheap_query.prepare(query_data,dim,32,ip);
    size_t hop_limit=std::numeric_limits<int>::max(),hop=0,nn_done=0;
    if(ef<k) { hop_limit=ef; ef=k; }
    Search_QuadHeap q0(ef,visited_list_pool_); auto q=&q0;
    const float entry_dist=getQueryDist(entry_point,query_data);
    if constexpr(CountDistances) ++ndc;
    q->push(entry_point,entry_dist,is_deleted(entry_point));
    q->set_visited(entry_point);
    while(!q->is_empty()) {
        if(hop>hop_limit) break;
        const auto current=q->get_next_id();
        const id_t cur=current.second;
        const float cur_dist=-current.first;
        if(cur_dist>q->get_dist_bound()) break;
        ++hop;
        std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
        auto [outs,sz,st]=getNeighbors(cur);
        size_t probed=0;
        std::vector<PendingJumpMid> pending;
        pending.reserve(sz);
        auto allows_mid=[&](float mid_dist) {
            const float bound=q->get_dist_bound();
            if(mid_dist-static_cast<double>(bound)>gate.mid_gap*BoundTwoHopGate::scale(bound)) return false;
            return probed<gate.max_mids_per_cur && nn_done<gate.max_nn_per_query;
        };
        auto probe_mid=[&](id_t mid) {
            ++probed;
            struct Candidate { id_t id; float dist; };
            std::vector<Candidate> candidates;
            std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
            auto [nn_outs,nn_sz,nn_st]=getNeighbors(mid);
            candidates.reserve(nn_sz);
            std::vector<JumpNNOption> pool;
            std::unordered_set<id_t> seen;
            for(int j=nn_st;j<nn_st+nn_sz;++j) {
                const id_t nn=nn_outs[j];
                if(q->is_visited(nn)) continue;
                // Deduplication is part of selection semantics, not tracing.
                if(!seen.insert(nn).second) continue;
                pool.push_back({nn,j,cheap_query.score(getData(nn)),0});
            }
            const auto indices=selectJumpNNIndices(pool,JumpNNSelection::CHEAP,8);
            std::vector<bool> chosen(nn_sz,false);
            for(int j:indices) chosen[j-nn_st]=true;
            // Keep the original physical scoring order, then the original
            // exact-distance sort before live beam admission.
            for(int j=nn_st;j<nn_st+nn_sz;++j) {
                if(!chosen[j-nn_st]) continue;
                if(nn_done>=gate.max_nn_per_query) break;
                const id_t nn=nn_outs[j];
                if(j<nn_st+nn_sz-1) {
                    q->prefetch_visited_list(nn_outs[j+1]);
                    __builtin_prefetch(getData(nn_outs[j+1]),0,1);
                }
                if(q->is_visited(nn)) continue;
                q->set_visited(nn);
                const float distance=getQueryDist(nn,query_data);
                ++nn_done; // Required by the algorithm's 4000 budget.
                if constexpr(CountDistances) ++ndc;
                candidates.push_back({nn,distance});
            }
            std::sort(candidates.begin(),candidates.end(),
                [](const Candidate& a,const Candidate& b) { return a.dist<b.dist; });
            for(const auto& candidate:candidates) {
                const bool admitted=q->top_candidates.size()<q->L || q->get_dist_bound()>candidate.dist;
                if(admitted) q->push(candidate.id,candidate.dist,is_deleted(candidate.id));
            }
        };
        for(int i=st;i<st+sz;++i) {
            const id_t mid=outs[i];
            if(i<st+sz-1) {
                q->prefetch_visited_list(outs[i+1]);
                __builtin_prefetch(getData(outs[i+1]),0,1);
            }
            if(q->is_visited(mid)) continue;
            q->set_visited(mid);
            const float mid_dist=getQueryDist(mid,query_data);
            if constexpr(CountDistances) ++ndc;
            const bool admitted=q->top_candidates.size()<q->L || q->get_dist_bound()>mid_dist;
            if(admitted) q->push(mid,mid_dist,is_deleted(mid));
            else pending.push_back({mid,mid_dist,i,0}); // Type unused by GAP selection.
        }
        std::vector<PendingJumpMid> eligible;
        for(const auto& mid:pending) if(allows_mid(mid.distance)) eligible.push_back(mid);
        // ALWAYS: state_topk/progress windows have no effect on this profile.
        if(!eligible.empty()) {
            const auto chosen=selectJumpMids(eligible,JumpMidSelection::GAP,2);
            for(size_t index:chosen) {
                const auto& mid=eligible[index];
                if(allows_mid(mid.distance)) probe_mid(mid.id);
            }
        }
    }
    q->releaseVisitedList();
    return q->get_result(k);
}
