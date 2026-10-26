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
struct Point { bool cheap=false; size_t ef=0; double recall=0; uint64_t expected_sum=0; };

int main(int argc,char** argv) {
    try {
        std::string index_path,query_path,gt_path,result_path,metric="ip_float";
        std::string beam_text="100,120,140,160,180,200,220,240,260,280,300,325,350,375,400,450,500,550,600,700,800,900,1000";
        std::string cheap_text="100,110,120,130,140,150,175,200,225,250,275,300,325,350,375,400,450,500,550,600";
        size_t k=100,offset=0,limit=10000,rounds=3,warmups=1,min_ms=300,seed=20261008,verify_limit=200;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--help") {
                std::cout << "Clean, single-worker Recall/QPS sweep. Fixed cheap32/NN8/gap.06/mid2/query4000/always.\n"
                    << "--index FILE --query FILE --gt FILE --out FILE --metric ip_float|l2_float\n"
                    << "--K 100 --query-offset 0 --query-limit 10000\n"
                    << "--beam-efs CSV --cheap-efs CSV --rounds 3 --warmup-passes 1 --min-batch-ms 300 --seed 20261008\n"
                    << "--verify-limit 200 (outside timing; all queries used for Recall and QPS)\n";
                return 0;
            }
            if(i+1>=argc) throw std::runtime_error("Missing value: "+arg);
            const std::string value=argv[++i];
            if(arg=="--index") index_path=value; else if(arg=="--query") query_path=value;
            else if(arg=="--gt") gt_path=value; else if(arg=="--out") result_path=value;
            else if(arg=="--metric") metric=value; else if(arg=="--K") k=integer(value);
            else if(arg=="--query-offset") offset=integer(value); else if(arg=="--query-limit") limit=integer(value);
            else if(arg=="--beam-efs") beam_text=value; else if(arg=="--cheap-efs") cheap_text=value;
            else if(arg=="--rounds") rounds=integer(value); else if(arg=="--warmup-passes") warmups=integer(value);
            else if(arg=="--min-batch-ms") min_ms=integer(value); else if(arg=="--seed") seed=integer(value);
            else if(arg=="--verify-limit") verify_limit=integer(value);
            else throw std::runtime_error("Unknown option: "+arg);
        }
        if(index_path.empty() || query_path.empty() || gt_path.empty() || result_path.empty() || !k || !rounds || !verify_limit || min_ms<10)
            throw std::runtime_error("Require paths, positive K/rounds and min-batch-ms>=10.");
        if(std::filesystem::exists(result_path)) throw std::runtime_error("Use a new output file.");
        auto beam_efs=ef_list(beam_text),cheap_efs=ef_list(cheap_text);
        if(beam_efs.front()<k || cheap_efs.front()<k) throw std::runtime_error("Require ef>=K.");
        const Metric distance=metric=="ip_float" ? IP_float : L2_float;
        if(metric!="ip_float" && metric!="l2_float") throw std::runtime_error("Only raw float IP/L2 supported.");
        auto queries=load_slice<float>(query_path,offset,limit);
        auto gt=load_slice<int32_t>(gt_path,offset,limit);
        if(queries.population!=gt.population || queries.count!=gt.count || gt.dim<k || queries.dim<32)
            throw std::runtime_error("Query/GT range, K or dimension mismatch.");
        for(float value:queries.values) {
            if((std::bit_cast<uint32_t>(value)&UINT32_C(0x7f800000))==UINT32_C(0x7f800000))
                throw std::runtime_error("Nonfinite query value.");
        }
        // Loading, validation, GT sets, reference comparisons and warmup are
        // all outside timed batches. One graph instance for both methods.
        HNSW_NGFix<float> graph(distance,index_path);
        if(graph.dim!=queries.dim || graph.n.load()<k || beam_efs.back()>graph.n.load() || cheap_efs.back()>graph.n.load())
            throw std::runtime_error("Index dimension/population incompatible with queries or ef.");
        graph.bound_twohop_gate=HNSW_NGFix<float>::clean32Profile();
        graph.jump_policy=JumpPolicy{};
        std::vector<std::unordered_set<id_t>> truth(queries.count);
        for(size_t i=0;i<queries.count;++i) for(size_t j=0;j<k;++j) {
            const int32_t id=gt.values[i*gt.dim+j];
            if(id<0 || size_t(id)>=graph.n.load() || !truth[i].insert(id_t(id)).second)
                throw std::runtime_error("Invalid/duplicate GT ID in top K.");
        }
        std::vector<Point> points;
        for(size_t ef:beam_efs) points.push_back({false,ef});
        for(size_t ef:cheap_efs) points.push_back({true,ef});
        const size_t verified=std::min(verify_limit,queries.count);
        std::cout << "Verifying " << verified << " queries at " << points.size() << " points; evaluating Recall on all "
            << queries.count << " queries (not timed).\n" << std::flush;
        for(auto& point:points) {
            size_t hits=0;
            for(size_t i=0;i<queries.count;++i) {
                float* query=queries.values.data()+i*queries.dim;
                size_t reference_ndc=0,counted_ndc=0,ignored=0;
                Result reference,counted,plain;
                if(point.cheap) {
                    if(i<verified) {
                        HNSW_NGFix<float>::TwoHopSearchStats stats;
                        reference=graph.searchKnnBoundTwoHop(query,k,point.ef,reference_ndc,&stats);
                        counted=graph.searchKnnCheap32Clean<true>(query,k,point.ef,counted_ndc);
                    }
                    plain=graph.searchKnnCheap32Clean<false>(query,k,point.ef,ignored);
                } else {
                    if(i<verified) {
                        reference=graph.searchKnn(query,k,point.ef,reference_ndc);
                        ++reference_ndc; // Legacy searchKnn omits its entry-point counter.
                        counted=graph.searchKnnBeamClean<true>(query,k,point.ef,counted_ndc);
                    }
                    plain=graph.searchKnnBeamClean<false>(query,k,point.ef,ignored);
                }
                if((i<verified && (reference!=counted || reference!=plain || counted_ndc!=reference_ndc)) || plain.size()!=k || ignored!=0)
                    throw std::runtime_error("Clean/reference mismatch: "+std::string(point.cheap?"cheap32_nn8":"beam")+
                        " ef="+std::to_string(point.ef)+" query="+std::to_string(offset+i));
                for(const auto& hit:plain) if(truth[i].count(hit.second)) ++hits;
                point.expected_sum+=fingerprint(plain);
            }
            point.recall=double(hits)/(double(queries.count)*k);
        }
        truth.clear(); truth.shrink_to_fit();
        gt.values.clear(); gt.values.shrink_to_fit();
        std::ofstream out(result_path);
        if(!out) throw std::runtime_error("Cannot open result file.");
        out << "method,ef,k,query_offset,queries,round,batch_queries,seconds,qps,recall,verified_queries,checksum\n";
        out << std::setprecision(17);
        auto batch=[&](const Point& point) {
            uint64_t sum=0;
            for(size_t i=0;i<queries.count;++i) {
                size_t ignored=0;
                auto result=point.cheap
                    ? graph.searchKnnCheap32Clean<false>(queries.values.data()+i*queries.dim,k,point.ef,ignored)
                    : graph.searchKnnBeamClean<false>(queries.values.data()+i*queries.dim,k,point.ef,ignored);
                sum+=fingerprint(result);
            }
            return sum;
        };
        std::mt19937_64 random(seed);
        std::vector<size_t> order(points.size()); std::iota(order.begin(),order.end(),0);
        for(size_t round=0;round<rounds;++round) {
            std::shuffle(order.begin(),order.end(),random);
            for(size_t index:order) {
                const auto& point=points[index];
                for(size_t pass=0;pass<warmups;++pass) {
                    if(batch(point)!=point.expected_sum) throw std::runtime_error("Warmup result checksum changed.");
                }
                uint64_t cycles=0,sum=0;
                const auto start=Clock::now();
                double seconds=0;
                do {
                    sum+=batch(point); ++cycles;
                    seconds=std::chrono::duration<double>(Clock::now()-start).count();
                } while(seconds*1000<double(min_ms));
                if(sum!=point.expected_sum*cycles) throw std::runtime_error("Timed result checksum changed.");
                const uint64_t batch_queries=cycles*queries.count;
                const double qps=batch_queries/seconds;
                // No CSV/console writes inside timed batches.
                out << (point.cheap?"cheap32_nn8":"beam") << ',' << point.ef << ',' << k << ',' << offset << ','
                    << queries.count << ',' << round << ',' << batch_queries << ',' << seconds << ',' << qps << ','
                    << point.recall << ',' << verified << ',' << sum << '\n';
                std::cout << "round=" << round << ' ' << (point.cheap?"cheap32_nn8":"beam") << " ef=" << point.ef
                    << " recall=" << point.recall << " QPS=" << qps << '\n' << std::flush;
            }
        }
        if(!out) throw std::runtime_error("Result write failed.");
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n'; return 1;
    }
}
