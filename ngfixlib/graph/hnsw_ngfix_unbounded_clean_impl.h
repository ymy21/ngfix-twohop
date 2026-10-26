// Included inside HNSW_NGFix<T>. Fixed default boundtwohop semantics:
// immediate, full NN scan, all edge types, no gates/budgets/policy.
// The caller validates this profile once outside timing. This kernel does
// not consult mutable gate settings and does not allocate attribution maps.
template<bool CountDistances=false>
std::vector<std::pair<float,id_t>> searchKnnBoundUnboundedClean(
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
        std::shared_lock<std::shared_mutex> cur_lock(node_locks[cur]);
        auto [outs,sz,st]=getNeighbors(cur);
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
            if(admitted) {
                q->push(mid,mid_dist,is_deleted(mid));
                continue;
            }
            struct Candidate { id_t id; float dist; };
            std::vector<Candidate> candidates;
            std::shared_lock<std::shared_mutex> mid_lock(node_locks[mid]);
            auto [nn_outs,nn_sz,nn_st]=getNeighbors(mid);
            candidates.reserve(nn_sz);
            for(int j=nn_st;j<nn_st+nn_sz;++j) {
                const id_t nn=nn_outs[j];
                if(j<nn_st+nn_sz-1) {
                    q->prefetch_visited_list(nn_outs[j+1]);
                    __builtin_prefetch(getData(nn_outs[j+1]),0,1);
                }
                if(q->is_visited(nn)) continue;
                q->set_visited(nn);
                const float dist=getQueryDist(nn,query_data);
                if constexpr(CountDistances) ++ndc;
                candidates.push_back({nn,dist});
            }
            // Preserve reference physical scoring order, distance-only sort
            // and live beam predicate. A rejected mid is never enqueued.
            std::sort(candidates.begin(),candidates.end(),
                [](const Candidate& a,const Candidate& b) { return a.dist<b.dist; });
            for(const auto& candidate:candidates) {
                if(q->top_candidates.size()<q->L || q->get_dist_bound()>candidate.dist)
                    q->push(candidate.id,candidate.dist,is_deleted(candidate.id));
            }
        }
    }
    q->releaseVisitedList();
    return q->get_result(k);
}
