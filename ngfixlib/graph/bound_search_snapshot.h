// Included inside HNSW_NGFix<T>. Diagnostic-only pre-probe snapshots.
// No ground truth or final search information is used to build these fields.
struct BoundDistanceSummary {
    double min=0,max=0,mean=0,variance=0,median=0,q25=0,q75=0;
};
struct BoundSearchSnapshot {
    size_t step=0,ndc=0,ndc_mid=0,ndc_nn=0,result_updates=0,topk_updates=0;
    size_t steps_since_topk_update=0,discovery_depth=0,frontier_size=0,beam_size=0,topk_size=0;
    double entry_dist=0;
    BoundDistanceSummary beam,topk;
};
static BoundDistanceSummary summarizeBoundDistances(const std::vector<double>& sorted) {
    BoundDistanceSummary s;
    if(sorted.empty()) return s;
    s.min=sorted.front();s.max=sorted.back();
    for(double d:sorted) s.mean+=d;
    s.mean/=sorted.size();
    for(double d:sorted) s.variance+=(d-s.mean)*(d-s.mean);
    s.variance/=sorted.size();
    auto quantile=[&](double fraction) {
        const double position=(sorted.size()-1)*fraction;
        const size_t low=static_cast<size_t>(position),high=std::min(low+1,sorted.size()-1);
        return sorted[low]+(sorted[high]-sorted[low])*(position-low);
    };
    s.median=quantile(.5);s.q25=quantile(.25);s.q75=quantile(.75);
    return s;
}
static BoundSearchSnapshot captureBoundSearchSnapshot(Search_QuadHeap& q,size_t k,size_t step,
        size_t ndc,size_t ndc_mid,size_t ndc_nn,size_t result_updates,size_t topk_updates,
        size_t last_topk_step,size_t discovery_depth,float entry_dist) {
    BoundSearchSnapshot s;
    s.step=step;s.ndc=ndc;s.ndc_mid=ndc_mid;s.ndc_nn=ndc_nn;
    s.result_updates=result_updates;s.topk_updates=topk_updates;
    s.steps_since_topk_update=step-last_topk_step;s.discovery_depth=discovery_depth;
    s.frontier_size=q.candidate_set.size();s.beam_size=q.top_candidates.size();s.entry_dist=entry_dist;
    std::vector<double> distances;
    distances.reserve(s.beam_size);
    // Copy only LIVE heap slots; do not call get_result(), which mutates the heap.
    for(size_t i=0;i<s.beam_size;++i) distances.push_back(q.top_candidates.data[i].first);
    std::sort(distances.begin(),distances.end());
    s.beam=summarizeBoundDistances(distances);
    distances.resize(std::min(k,distances.size()));s.topk_size=distances.size();
    s.topk=summarizeBoundDistances(distances);
    return s;
}
static const char* boundSearchSnapshotHeader() {
    return "state_step,state_ndc,state_ndc_mid,state_ndc_nn,state_result_updates,state_topk_updates,"
           "state_steps_since_topk_update,state_discovery_depth,state_frontier_size,state_beam_size,state_topk_size,"
           "state_entry_dist,beam_min,beam_max,beam_mean,beam_variance,beam_median,beam_q25,beam_q75,"
           "topk_min,topk_max,topk_mean,topk_variance,topk_median,topk_q25,topk_q75";
}
static void writeBoundSearchSnapshot(std::ostream& out,const BoundSearchSnapshot& s) {
    out<<s.step<<','<<s.ndc<<','<<s.ndc_mid<<','<<s.ndc_nn<<','<<s.result_updates<<','<<s.topk_updates
       <<','<<s.steps_since_topk_update<<','<<s.discovery_depth<<','<<s.frontier_size<<','<<s.beam_size
       <<','<<s.topk_size<<','<<s.entry_dist;
    for(const auto& d:{s.beam,s.topk})
        out<<','<<d.min<<','<<d.max<<','<<d.mean<<','<<d.variance<<','<<d.median<<','<<d.q25<<','<<d.q75;
}
