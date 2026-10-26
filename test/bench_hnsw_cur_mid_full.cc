// New benchmark: legacy unbounded benchmark remains unchanged.
#include "ngfixlib/graph/hnsw_ngfix.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_set>
#include <sys/time.h>
using namespace ngfixlib;
using Clock=std::chrono::steady_clock;

static size_t integer(const std::string& text) {
    if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer: "+text);
    return std::stoull(text);
}
static std::vector<size_t> ef_list(const std::string& text) {
    std::vector<size_t> values; std::stringstream ss(text); std::string word;
    while(std::getline(ss,word,',')) values.push_back(integer(word));
    std::sort(values.begin(),values.end());
    if(values.empty() || !values.front() || std::adjacent_find(values.begin(),values.end())!=values.end())
        throw std::runtime_error("Require distinct positive ef values.");
    return values;
}
template<class Value> struct Slice { size_t population=0,dim=0,count=0; std::vector<Value> values; };
template<class Value> static Slice<Value> load_slice(const std::string& path,size_t offset,size_t limit) {
    std::ifstream in(path,std::ios::binary); uint32_t n=0,d=0;
    if(!in.read(reinterpret_cast<char*>(&n),4) || !in.read(reinterpret_cast<char*>(&d),4) || !n || !d)
        throw std::runtime_error("Invalid vector header: "+path);
    if(offset>=n || !limit || limit>size_t(n)-offset) throw std::runtime_error("Query range outside file: "+path);
    if(uint64_t(n)*d>(std::numeric_limits<uint64_t>::max()-8)/sizeof(Value))
        throw std::runtime_error("Vector payload size overflow: "+path);
    const uint64_t payload=uint64_t(n)*d*sizeof(Value);
    if(std::filesystem::file_size(path)<8+payload) throw std::runtime_error("Truncated vector file: "+path);
    Slice<Value> s; s.population=n; s.dim=d; s.count=limit;
    s.values.resize(s.count*s.dim);
    in.seekg(8+uint64_t(offset)*d*sizeof(Value));
    if(!in.read(reinterpret_cast<char*>(s.values.data()),s.values.size()*sizeof(Value)))
        throw std::runtime_error("Cannot read vector slice: "+path);
    return s;
}
using Result=std::vector<std::pair<float,id_t>>;
static uint64_t fingerprint(const Result& r) {
    // Constant work per returned result, observable after timing. No GT lookup,
    // result comparison, per-neighbor instrumentation or per-query timestamp.
    return uint64_t(r.front().second)+uint64_t(r.back().second)*UINT64_C(0x9e3779b97f4a7c15)+r.size();
}
struct Point { bool cheap=false; size_t ef=0; double recall=0,mean_ndc=0; uint64_t expected_sum=0; };

int main(int argc,char** argv) {
    try {
        std::string index_path,query_path,gt_path,result_path,metric="ip_float";
        std::string marks_path; ABMode ab_mode=ABMode::NONE;
        std::string timing_protocol="batch";
        double mid_gap=0.0266,cur_margin=0.0127;
        size_t mid_cap=0; bool beam_only=false;
        double detour_ratio=std::numeric_limits<double>::max();
        std::string beam_text="100,120,140,160,180,200,220,240,260,280,300,325,350,375,400,450,500,550,600,700,800,900,1000";
        std::string cheap_text=beam_text;
        size_t k=100,offset=0,limit=10000,rounds=3,warmups=1,min_ms=300,seed=20261008,verify_limit=200; bool require_unrepaired=true;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") {
                std::cout << "Clean, single-worker Recall/QPS sweep. Cur/mid gates full NN after one-hop; configurable cur/mid distance gates; no diagnostic work in timing.\n"
                    << "--index FILE --query FILE --gt FILE --out FILE --metric ip_float|l2_float\n"
                    << "--K 100 --query-offset 0 --query-limit 10000\n"
                    << "--beam-efs CSV --bound-efs CSV --rounds 3 --warmup-passes 1 --min-batch-ms 300 --seed 20261008\n"
                    << "--verify-limit 200 --require-unrepaired 1 (set 0 only for NGFix reference runs)\n";
                std::cout << "--mid-gap 0.0266 --cur-margin 0.0127 --mid-cap 0 --detour-ratio MAX --beam-only 0 (no NN budgets)\n";
                std::cout << "--timing-protocol batch|original (original: gettimeofday around each search, NDC inside)\n";
                std::cout << "--marks FILE --ab-mode A|B (optional offline labels; selected mids retain FULL NN)\n";
                return 0;
            }
            if(i+1>=argc) throw std::runtime_error("Missing value: "+arg);
            const std::string value=argv[++i];
            if(arg=="--index") index_path=value; else if(arg=="--query") query_path=value;
            else if(arg=="--gt") gt_path=value; else if(arg=="--out") result_path=value;
            else if(arg=="--metric") metric=value; else if(arg=="--K") k=integer(value);
            else if(arg=="--query-offset") offset=integer(value); else if(arg=="--query-limit") limit=integer(value);
            else if(arg=="--beam-efs") beam_text=value; else if(arg=="--bound-efs") cheap_text=value;
            else if(arg=="--rounds") rounds=integer(value); else if(arg=="--warmup-passes") warmups=integer(value);
            else if(arg=="--min-batch-ms") min_ms=integer(value); else if(arg=="--seed") seed=integer(value);
            else if(arg=="--verify-limit") verify_limit=integer(value);
            else if(arg=="--require-unrepaired") require_unrepaired=integer(value)!=0;
            else if(arg=="--timing-protocol") timing_protocol=value;
            else if(arg=="--mid-gap" || arg=="--cur-margin" || arg=="--detour-ratio") {
                size_t consumed=0; const double x=std::stod(value,&consumed);
                if(consumed!=value.size()) throw std::runtime_error("Invalid floating option");
                if(arg=="--mid-gap") mid_gap=x; else if(arg=="--cur-margin") cur_margin=x; else detour_ratio=x;
            }
            else if(arg=="--mid-cap") mid_cap=integer(value);
            else if(arg=="--beam-only") beam_only=integer(value)!=0;
            else if(arg=="--marks") marks_path=value;
            else if(arg=="--ab-mode") ab_mode=parseABMode(value);
            else throw std::runtime_error("Unknown option: "+arg);
        }
        if(index_path.empty() || query_path.empty() || gt_path.empty() || result_path.empty() || !k || !rounds || !verify_limit || min_ms<10)
            throw std::runtime_error("Require paths, positive K/rounds and min-batch-ms>=10.");
        if(timing_protocol!="batch" && timing_protocol!="original") throw std::runtime_error("Unknown timing protocol");
        if(marks_path.empty()!=(ab_mode==ABMode::NONE)) throw std::runtime_error("Specify both --marks and --ab-mode A|B");
        if(ab_mode!=ABMode::NONE && (beam_only || metric!="ip_float" || !require_unrepaired))
            throw std::runtime_error("AB stage 1 requires unrepaired IP graph and bound evaluation");
        const std::string bound_name=ab_mode==ABMode::A?"ab_A_full":(ab_mode==ABMode::B?"ab_B_full":"cur_mid_full");
        if(timing_protocol=="original" && warmups) throw std::runtime_error("Original protocol requires --warmup-passes 0");
        if(std::filesystem::exists(result_path) || std::filesystem::exists(result_path+".queries.csv") ||
           std::filesystem::exists(result_path+".latencies.csv")) throw std::runtime_error("Use a new output file.");
        auto beam_efs=ef_list(beam_text),cheap_efs=ef_list(cheap_text);
        if(beam_efs.front()<k || cheap_efs.front()<k) throw std::runtime_error("Require ef>=K.");
        const Metric distance=metric=="ip_float" ? IP_float : L2_float;
        if(metric!="ip_float" && metric!="l2_float") throw std::runtime_error("Only raw float IP/L2 supported.");
        auto queries=load_slice<float>(query_path,offset,limit);
        auto gt=load_slice<int32_t>(gt_path,offset,limit);
        if(queries.population!=gt.population || queries.count!=gt.count || gt.dim<k)
            throw std::runtime_error("Query/GT range, K or dimension mismatch.");
        for(float value:queries.values) {
            if((std::bit_cast<uint32_t>(value)&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))
                throw std::runtime_error("Nonfinite query value.");
        }
        // Loading, validation, GT sets, reference comparisons and warmup are
        // all outside timed batches. One graph instance for both methods.
        if(ab_mode!=ABMode::NONE) validateABIndex(index_path);
        HNSW_NGFix<float> graph(distance,index_path);
        if(graph.dim!=queries.dim || graph.n.load()<k || beam_efs.back()>graph.n.load() || cheap_efs.back()>graph.n.load())
            throw std::runtime_error("Index dimension/population incompatible with queries or ef.");
        graph.bound_twohop_gate=HNSW_NGFix<float>::curMidFullProfile(mid_gap,cur_margin,detour_ratio,mid_cap);
        graph.jump_policy=JumpPolicy{};
        // Graph audit is outside timing, including actual degree and entry.
        graph.printGraphInfo();
        if(require_unrepaired) for(size_t i=0;i<graph.n.load();++i)
            if(GET_NGFIX_SZ(reinterpret_cast<uint8_t*>(graph.Graph[i].neighbors)))
                throw std::runtime_error("Stage 1 requires an unrepaired index.");
        ABMarks marks;
        if(ab_mode!=ABMode::NONE) {
            marks=ABMarks::load(marks_path,ABMarks::layout(graph));
            std::cout<<"AB labels: L="<<marks.top_l<<" train_queries="<<marks.queries
                     <<" bytes="<<marks.flags.size()+marks.offsets.size()*8<<'\n';
        }
        auto bound_counted=[&](float* q,size_t ef,size_t& ndc,HNSW_NGFix<float>::CurMidFullCounters* c=nullptr) {
            return ab_mode==ABMode::NONE ? graph.searchKnnCurMidFullClean<true>(q,k,ef,ndc,c)
                : graph.searchKnnCurMidFullClean<true,true>(q,k,ef,ndc,c,&marks,ab_mode);
        };
        auto bound_plain=[&](float* q,size_t ef,size_t& ndc) {
            return ab_mode==ABMode::NONE ? graph.searchKnnCurMidFullClean<false>(q,k,ef,ndc)
                : graph.searchKnnCurMidFullClean<false,true>(q,k,ef,ndc,nullptr,&marks,ab_mode);
        };
        std::ofstream per_query(result_path+".queries.csv");
        if(!per_query) throw std::runtime_error("Cannot create query statistics");
        per_query << "method,ef,query_id,recall,ndc,ndc_entry,ndc_mid,ndc_nn,rejected_mids,blocked_cur,blocked_gap,blocked_ratio,blocked_selection,probes,successful_probes,admitted_nn,scanned_nn,expanded,blocked_mark,mark_seen,mark_unknown,mark_allowed,nn_zero_admission\n" << std::setprecision(17);
        std::vector<std::unordered_set<id_t>> truth(queries.count);
        for(size_t i=0;i<queries.count;++i) for(size_t j=0;j<k;++j) {
            const int32_t id=gt.values[i*gt.dim+j];
            if(id<0 || size_t(id)>=graph.n.load() || !truth[i].insert(id_t(id)).second)
                throw std::runtime_error("Invalid/duplicate GT ID in top K.");
        }
        std::vector<Point> points;
        for(size_t ef:beam_efs) points.push_back({false,ef});
        if(!beam_only) for(size_t ef:cheap_efs) points.push_back({true,ef});
        const size_t verified=std::min(verify_limit,queries.count);
        std::cout << "Verifying " << verified << " queries at " << points.size() << " points; evaluating Recall on all "
            << queries.count << " queries (not timed).\n" << std::flush;
        for(auto& point:points) {
            size_t hits=0; uint64_t total_ndc=0;
            for(size_t i=0;i<queries.count;++i) {
                float* query=queries.values.data()+i*queries.dim;
                size_t reference_ndc=0,counted_ndc=0,ignored=0;
                Result reference,counted,plain;
                if(point.cheap) {
                    if(i<verified && ab_mode==ABMode::NONE) {
                        HNSW_NGFix<float>::TwoHopSearchStats stats;
                        reference=graph.searchKnnBoundTwoHop(query,k,point.ef,reference_ndc,&stats);
                        
                    }
                    plain=bound_plain(query,point.ef,ignored);
                } else {
                    if(i<verified) {
                        reference=graph.searchKnn(query,k,point.ef,reference_ndc);
                        ++reference_ndc; // Legacy searchKnn omits its entry-point counter.
                        
                    }
                    plain=graph.searchKnnBeamClean<false>(query,k,point.ef,ignored);
                }
                HNSW_NGFix<float>::CurMidFullCounters counters;
                counted=point.cheap
                    ? bound_counted(query,point.ef,counted_ndc,&counters)
                    : graph.searchKnnBeamClean<true>(query,k,point.ef,counted_ndc);
                total_ndc+=counted_ndc;
                if(counted!=plain) throw std::runtime_error("Counted/plain mismatch");
                if((i<verified && (!point.cheap || ab_mode==ABMode::NONE) && (reference!=counted || reference!=plain || counted_ndc!=reference_ndc)) || plain.size()!=k || ignored!=0)
                    throw std::runtime_error("Clean/reference mismatch: "+std::string(point.cheap?bound_name:"beam")+
                        " ef="+std::to_string(point.ef)+" query="+std::to_string(offset+i));
                size_t query_hits=0;
                for(const auto& hit:plain) if(truth[i].count(hit.second)) ++query_hits;
                hits+=query_hits;
                per_query << (point.cheap?bound_name:"beam") << ',' << point.ef << ','
                    << offset+i << ',' << double(query_hits)/k << ',' << counted_ndc
                    << ',' << (point.cheap?counters.entry:1) << ',' << (point.cheap?counters.mid:counted_ndc-1)
                    << ',' << counters.nn << ',' << counters.rejected << ',' << counters.blocked_cur
                    << ',' << counters.blocked_gap << ',' << counters.blocked_ratio << ',' << counters.blocked_selection
                    << ',' << counters.probes << ',' << counters.successful_probes << ',' << counters.admitted_nn
                    << ',' << counters.scanned_nn << ',' << counters.expanded << ',' << counters.blocked_mark
                    << ',' << counters.mark_seen << ',' << counters.mark_unknown << ',' << counters.mark_allowed
                    << ',' << counters.nn_zero_admission << '\n';
                point.expected_sum+=fingerprint(plain);
            }
            point.recall=double(hits)/(double(queries.count)*k);
            point.mean_ndc=double(total_ndc)/queries.count;
            std::cout << "evaluation " << (point.cheap?bound_name:"beam") << " ef=" << point.ef
                      << " recall=" << point.recall << " mean_ndc=" << point.mean_ndc << '\n' << std::flush;
        }
        if(!per_query) throw std::runtime_error("Query statistics write failed");
        per_query.close();
        truth.clear(); truth.shrink_to_fit();
        if(timing_protocol=="batch") { gt.values.clear(); gt.values.shrink_to_fit(); }
        std::ofstream out(result_path);
        if(!out) throw std::runtime_error("Cannot open result file.");
        out << "method,ef,k,query_offset,queries,round,batch_queries,seconds,qps,recall,verified_queries,checksum,mean_ndc,timing_protocol\n";
        out << std::setprecision(17);
        std::ofstream latency_out;
        if(timing_protocol=="original") {
            latency_out.open(result_path+".latencies.csv");
            if(!latency_out) throw std::runtime_error("Cannot open per-query latency file");
            latency_out << "method,ef,round,query_id,latency_us,ndc\n";
        }
        auto batch=[&](const Point& point) {
            uint64_t sum=0;
            for(size_t i=0;i<queries.count;++i) {
                size_t ignored=0;
                auto result=point.cheap
                    ? bound_plain(queries.values.data()+i*queries.dim,point.ef,ignored)
                    : graph.searchKnnBeamClean<false>(queries.values.data()+i*queries.dim,k,point.ef,ignored);
                sum+=fingerprint(result);
            }
            return sum;
        };
        std::mt19937_64 random(seed);
        std::vector<size_t> order(points.size()); std::iota(order.begin(),order.end(),0);
        for(size_t round=0;round<rounds;++round) {
            if(timing_protocol=="batch") std::shuffle(order.begin(),order.end(),random);
            for(size_t index:order) {
                const auto& point=points[index];
                for(size_t pass=0;pass<warmups;++pass) {
                    if(batch(point)!=point.expected_sum) throw std::runtime_error("Warmup result checksum changed.");
                }
                uint64_t cycles=0,sum=0;
                double seconds=0;
                if(timing_protocol=="batch") {
                    const auto start=Clock::now();
                    do {
                        sum+=batch(point); ++cycles;
                        seconds=std::chrono::duration<double>(Clock::now()-start).count();
                    } while(seconds*1000<double(min_ms));
                } else {
                    // Match upstream TestSingleQuery's timing boundary and clock.
                    // Baseline calls the unchanged upstream searchKnn directly.
                    // Bound uses its equivalent counted kernel, without trace maps.
                    uint64_t ndc_sum=0,hits=0;
                    int64_t total_us=0;
                    std::vector<int64_t> latencies(queries.count);
                    std::vector<size_t> distances(queries.count);
                    double ratio_sum=0;
                    for(size_t i=0;i<queries.count;++i) {
                        float* query=queries.values.data()+i*queries.dim;
                        size_t ndc=0;
                        timeval start,end;
                        gettimeofday(&start,nullptr);
                        auto result=point.cheap
                            ? bound_counted(query,point.ef,ndc)
                            : graph.searchKnn(query,k,point.ef,ndc);
                        gettimeofday(&end,nullptr);
                        const int64_t us=(int64_t(end.tv_sec)*1000000+end.tv_usec)
                                       -(int64_t(start.tv_sec)*1000000+start.tv_usec);
                        if(us<0) throw std::runtime_error("Wall clock moved backwards");
                        // Entry normalization occurs AFTER timing. The upstream
                        // baseline counts only freshly scored adjacency targets.
                        if(!point.cheap) ++ndc;
                        distances[i]=ndc;latencies[i]=us;total_us+=us;ndc_sum+=ndc;
                        sum+=fingerprint(result);
                        // Reproduce upstream's between-query GT evaluation.
                        std::unordered_set<id_t> gtset;
                        for(size_t j=0;j<k;++j) gtset.insert(gt.values[i*gt.dim+j]);
                        for(const auto& hit:result) if(gtset.erase(hit.second)) ++hits;
                        for(size_t j=0;j<k;++j) {
                            const float d0=graph.getDist(result[j].second,query);
                            const float d1=graph.getDist(gt.values[i*gt.dim+j],query);
                            if(std::abs(d1)>=.00001f) ratio_sum+=d0/d1;
                        }
                    }
                    if(total_us<=0 || std::abs(double(ndc_sum)/queries.count-point.mean_ndc)>1e-9 ||
                       std::abs(double(hits)/(queries.count*k)-point.recall)>1e-12)
                        throw std::runtime_error("Original timing results/NDC mismatch");
                    seconds=double(total_us)/1000000;cycles=1;
                    for(size_t i=0;i<queries.count;++i)
                        latency_out << (point.cheap?bound_name:"beam") << ',' << point.ef << ',' << round
                                    << ',' << offset+i << ',' << latencies[i] << ',' << distances[i] << '\n';
                    std::cout << "evaluation-only mean distance ratio=" << ratio_sum/(queries.count*k) << '\n';
                }
                if(sum!=point.expected_sum*cycles) throw std::runtime_error("Timed result checksum changed.");
                const uint64_t batch_queries=cycles*queries.count;
                const double qps=batch_queries/seconds;
                // No CSV/console writes inside timed batches.
                out << (point.cheap?bound_name:"beam") << ',' << point.ef << ',' << k << ',' << offset << ','
                    << queries.count << ',' << round << ',' << batch_queries << ',' << seconds << ',' << qps << ','
                    << point.recall << ',' << ((point.cheap && ab_mode!=ABMode::NONE)?0:verified) << ',' << sum << ',' << point.mean_ndc << ',' << timing_protocol << '\n';
                std::cout << "round=" << round << ' ' << (point.cheap?bound_name:"beam") << " ef=" << point.ef
                    << " recall=" << point.recall << " QPS=" << qps << '\n' << std::flush;
            }
        }
        if(!out) throw std::runtime_error("Result write failed.");
        if(timing_protocol=="original" && !latency_out) throw std::runtime_error("Latency write failed");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n'; return 1;
    }
}
