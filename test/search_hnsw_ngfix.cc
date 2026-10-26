#include "ngfixlib/graph/hnsw_ngfix.h"
#include "tools/data_loader.h"
#include "tools/result_evaluation.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ngfixlib;

static std::string StripCsvExtension(const std::string& path) {
    if(path.size() >= 4 && path.substr(path.size() - 4) == ".csv")
        return path.substr(0, path.size() - 4);
    return path;
}

static std::vector<int> ParseIntCsv(const std::string& text) {
    std::vector<int> values;
    std::stringstream ss(text);
    std::string token;
    while(std::getline(ss, token, ',')) {
        if(token.empty()) continue;
        const int value = std::stoi(token);
        if(value <= 0) throw std::runtime_error("ef values must be positive");
        values.push_back(value);
    }
    return values;
}

static std::vector<float> ParseFloatCsv(const std::string& text) {
    std::vector<float> values;
    std::stringstream ss(text);
    std::string token;
    while(std::getline(ss, token, ',')) {
        if(token.empty()) continue;
        const float value = std::stof(token);
        if(value < 0.0f) throw std::runtime_error("DABS gamma must be non-negative");
        values.push_back(value);
    }
    return values;
}

struct NormSummary {
    size_t population = 0;
    size_t sampled = 0;
    size_t finite = 0;
    size_t nonfinite = 0;
    size_t outside_tolerance = 0;
    double min_norm = std::numeric_limits<double>::infinity();
    double max_norm = -std::numeric_limits<double>::infinity();
    double sum_norm = 0.0;

    double meanNorm() const {
        return finite ? sum_norm / static_cast<double>(finite)
                      : std::numeric_limits<double>::quiet_NaN();
    }
    bool unitNormalized() const {
        return sampled > 0 && nonfinite == 0 && outside_tolerance == 0;
    }
};

template<typename VectorAccessor>
static NormSummary SummarizeNorms(size_t population,
                                  size_t sample_limit,
                                  size_t dim,
                                  double tolerance,
                                  VectorAccessor vector_at)
{
    NormSummary summary;
    summary.population = population;
    summary.sampled = sample_limit == 0 ? population : std::min(population, sample_limit);

    for(size_t i = 0; i < summary.sampled; ++i) {
        const size_t index = summary.sampled == population
            ? i
            : static_cast<size_t>(
                (static_cast<long double>(i) * population) / summary.sampled);
        const float* vector = vector_at(index);
        long double squared_norm = 0.0L;
        for(size_t j = 0; j < dim; ++j) {
            const long double value = static_cast<long double>(vector[j]);
            squared_norm += value * value;
        }
        const double norm = std::sqrt(static_cast<double>(squared_norm));
        if(!std::isfinite(norm)) {
            ++summary.nonfinite;
            continue;
        }
        ++summary.finite;
        summary.sum_norm += norm;
        summary.min_norm = std::min(summary.min_norm, norm);
        summary.max_norm = std::max(summary.max_norm, norm);
        if(std::abs(norm - 1.0) > tolerance) ++summary.outside_tolerance;
    }
    return summary;
}

static void WriteNormSummary(std::ostream& out,
                             const char* source,
                             const NormSummary& summary,
                             double tolerance)
{
    const double min_norm = summary.finite
        ? summary.min_norm : std::numeric_limits<double>::quiet_NaN();
    const double max_norm = summary.finite
        ? summary.max_norm : std::numeric_limits<double>::quiet_NaN();
    out << source << "," << summary.population << "," << summary.sampled << ","
        << summary.meanNorm() << "," << min_norm << "," << max_norm << ","
        << summary.outside_tolerance << "," << summary.nonfinite << ","
        << tolerance << "," << (summary.unitNormalized() ? 1 : 0) << "\n";
}

int main(int argc, char* argv[])
{
    int k = 0;
    std::unordered_map<std::string,std::string> paths;

    std::string search_mode = "baseline";
    std::string history_path = "";
    float history_threshold = 0.05;
    BoundTwoHopGate jump_gate;
    JumpPolicy jump_policy;
    std::string jump_tree_path, jump_history_path;
    bool event_trace_compact = false;
    bool nn_trace = false;
    bool trace_search_state = false;
    int event_trace_ef = -1;
    size_t event_trace_query_limit = std::numeric_limits<size_t>::max();
    size_t query_limit = std::numeric_limits<size_t>::max();
    size_t query_offset = 0;
    std::vector<int> ef_values;
    std::vector<float> dabs_gammas = {0.0f, 0.02f, 0.05f, 0.1f, 0.2f, 0.4f};

    bool check_norms = true;
    bool allow_nonunit_dabs = false;
    size_t norm_sample_size = 10000;
    double norm_tolerance = 1e-3;
    bool write_graph_audit = true;

    // Structural audit can run on an existing index without loading test queries.
    bool graph_audit_only = false;
    std::string audit_prefix = "";

    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if(arg == "--help") {
            std::cout << "Bound-two-hop gates (use boundtwohop or boundtwohopcmp; all disabled by default):\n"
                      << "  --trace_search_state 0|1  diagnostic pre-probe state sidecar; requires --event_trace_ef\n"
                      << "  --jump_cur_topk N           cur among best N scored live nodes (0 disables)\n"
                      << "  --jump_min_hops N           first allowed expansion ordinal\n"
                      << "  --jump_cur_margin X         min (bound-cur)/max(abs(bound),1e-6)\n"
                      << "  --jump_mid_gap X            max (mid-bound)/max(abs(bound),1e-6)\n"
                      << "  --jump_detour_ratio X       require bound>cur and mid-bound <= X*(bound-cur)\n"
                      << "  --jump_max_mids_per_cur N   probe cap; selected after_onehop (0 unlimited)\n"
                      << "  --jump_max_nn_per_query N   second-hop NDC cap (0 unlimited)\n"
                      << "  --jump_schedule MODE      immediate|after_onehop (default immediate)\n"
                      << "  --jump_mid_selection MODE adjacency|gap|stratified; after_onehop only\n"
                      << "  --jump_trigger MODE       always|stall|stall_frontier; after_onehop only\n"
                      << "  --jump_stall_window N     completed cur windows without ordinary top-K changes\n"
                      << "  --jump_max_nn_per_mid N   fresh second-hop scores per mid (0 unlimited)\n"
                      << "  --jump_max_nn_per_cur N   fresh second-hop scores per cur batch (0 unlimited)\n"
                      << "  --jump_first_edge FILTER    all/base/ngfix/ngfix_finite/ngfix_inf/rfix\n"
                      << "  --jump_second_edge FILTER   same choices; filter before scoring\n"
                      << "  --jump_policy MODE          off/tree/history/hybrid (same boundtwohop kernel)\n"
                      << "  --jump_tree_path FILE       exported V1 shallow tree\n"
                      << "  --jump_tree_threshold X     minimum predicted useful probes per NDC\n"
                      << "  --jump_history_path FILE    exported V1 history table, same index only\n"
                      << "  --jump_history_threshold X  minimum historical useful probes per NDC\n"
                      << "  --jump_history_min_queries N  minimum distinct training queries (default 3)\n"
                      << "  --jump_history_unknown allow|deny  history-only fallback (default allow)\n"
                      << "  --query_offset N            first input row, preserves global query IDs\n"
                      << "  --jump_nn_selection full|prefix|random|cheap\n"
                      << "  --jump_nn_limit 8  --jump_nn_cheap_dims 16  --jump_nn_seed 17\n"
                      << "  --nn_trace 1  all fresh NN candidates at event_trace_ef\n"
                      << "  --event_trace_compact 1     emit compact pre-probe features and labels\n";
            return 0;
        }
        if(i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);

        auto jump_size = [&]() -> size_t {
            const std::string value = argv[++i];
            if(value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error(arg + " requires a non-negative integer");
            return static_cast<size_t>(std::stoull(value));
        };
        auto jump_float = [&]() -> double {
            const std::string value = argv[++i];
            size_t consumed = 0;
            const double result = std::stod(value, &consumed);
            if(consumed != value.size() || !BoundTwoHopGate::finite_value(result) || result < 0)
                throw std::runtime_error(arg + " requires a finite non-negative number");
            return result;
        };

        if(arg=="--test_query_path") paths["test_query_path"]=argv[++i];
        else if(arg=="--test_gt_path") paths["test_gt_path"]=argv[++i];
        else if(arg=="--metric") paths["metric"]=argv[++i];
        else if(arg=="--index_path") paths["index_path"]=argv[++i];
        else if(arg=="--result_path") paths["result_path"]=argv[++i];
        else if(arg=="--K") k=std::stoi(argv[++i]);

        else if(arg=="--search_mode")
            search_mode = argv[++i];

        else if(arg=="--history_path")
            history_path = argv[++i];

        else if(arg=="--history_threshold")
            history_threshold = std::stof(argv[++i]);

        else if(arg=="--jump_cur_topk") jump_gate.cur_topk = jump_size();
        else if(arg=="--jump_min_hops") jump_gate.min_hops = jump_size();
        else if(arg=="--jump_cur_margin") jump_gate.cur_margin = jump_float();
        else if(arg=="--jump_mid_gap") jump_gate.mid_gap = jump_float();
        else if(arg=="--jump_detour_ratio") jump_gate.detour_ratio = jump_float();
        else if(arg=="--jump_max_mids_per_cur") jump_gate.max_mids_per_cur = jump_size();
        else if(arg=="--jump_max_nn_per_query") jump_gate.max_nn_per_query = jump_size();
        else if(arg=="--jump_schedule") jump_gate.schedule = parseJumpSchedule(argv[++i]);
        else if(arg=="--jump_mid_selection") jump_gate.selection = parseJumpMidSelection(argv[++i]);
        else if(arg=="--jump_trigger") jump_gate.trigger = parseJumpTrigger(argv[++i]);
        else if(arg=="--jump_stall_window") jump_gate.stall_window = jump_size();
        else if(arg=="--jump_max_nn_per_mid") jump_gate.max_nn_per_mid = jump_size();
        else if(arg=="--jump_max_nn_per_cur") jump_gate.max_nn_per_cur = jump_size();
        else if(arg=="--jump_first_edge") jump_gate.first_edge = parseJumpEdgeFilter(argv[++i]);
        else if(arg=="--jump_second_edge") jump_gate.second_edge = parseJumpEdgeFilter(argv[++i]);
        else if(arg=="--jump_policy") jump_policy.mode = parseJumpPolicy(argv[++i]);
        else if(arg=="--jump_tree_path") jump_tree_path = argv[++i];
        else if(arg=="--jump_history_path") jump_history_path = argv[++i];
        else if(arg=="--jump_tree_threshold") jump_policy.tree_threshold = jump_float();
        else if(arg=="--jump_history_threshold") jump_policy.history_threshold = jump_float();
        else if(arg=="--jump_history_min_queries") jump_policy.history_min_queries = jump_size();
        else if(arg=="--jump_history_unknown") {
            const std::string value = argv[++i];
            if(value != "allow" && value != "deny")
                throw std::runtime_error("--jump_history_unknown requires allow or deny");
            jump_policy.history_unknown_allow = value == "allow";
        }
        else if(arg=="--query_offset") query_offset = jump_size();
        else if(arg=="--jump_nn_selection") jump_gate.nn_selection = parseJumpNNSelection(argv[++i]);
        else if(arg=="--jump_nn_limit") jump_gate.nn_limit = jump_size();
        else if(arg=="--jump_nn_cheap_dims") jump_gate.nn_cheap_dims = jump_size();
        else if(arg=="--jump_nn_seed") jump_gate.nn_seed = jump_size();
        else if(arg=="--nn_trace") nn_trace = jump_size() != 0;
        else if(arg=="--trace_search_state") trace_search_state = jump_size() != 0;
        else if(arg=="--event_trace_compact") event_trace_compact = jump_size() != 0;

        else if(arg=="--event_trace_ef")
            event_trace_ef = std::stoi(argv[++i]);

        else if(arg=="--event_trace_query_limit")
            event_trace_query_limit = static_cast<size_t>(std::stoull(argv[++i]));

        else if(arg=="--query_limit")
            query_limit = static_cast<size_t>(std::stoull(argv[++i]));

        else if(arg=="--ef_values")
            ef_values = ParseIntCsv(argv[++i]);

        else if(arg=="--dabs_gammas")
            dabs_gammas = ParseFloatCsv(argv[++i]);

        else if(arg=="--check_norms")
            check_norms = (std::stoi(argv[++i]) != 0);

        else if(arg=="--allow_nonunit_dabs")
            allow_nonunit_dabs = (std::stoi(argv[++i]) != 0);

        else if(arg=="--norm_sample_size")
            norm_sample_size = static_cast<size_t>(std::stoull(argv[++i]));

        else if(arg=="--norm_tolerance")
            norm_tolerance = std::stod(argv[++i]);

        else if(arg=="--write_graph_audit")
            write_graph_audit = (std::stoi(argv[++i]) != 0);

        else if(arg=="--graph_audit_only")
            graph_audit_only = (std::stoi(argv[++i]) != 0);

        else if(arg=="--audit_prefix")
            audit_prefix = argv[++i];
        else throw std::runtime_error("Unknown option: " + arg);
    }

    jump_gate.validate();
    if(nn_trace && event_trace_ef < 0) throw std::runtime_error("--nn_trace requires --event_trace_ef.");
    if(trace_search_state && event_trace_ef < 0) throw std::runtime_error("--trace_search_state requires --event_trace_ef.");
    if((jump_gate.active() || jump_policy.active()) && search_mode != "boundtwohop" && search_mode != "boundtwohopcmp")
        throw std::runtime_error("Jump gates require boundtwohop or boundtwohopcmp.");
    if((!jump_tree_path.empty() && !jump_policy.usesTree()) ||
       (!jump_history_path.empty() && !jump_policy.usesHistory()))
        throw std::runtime_error("Set --jump_policy to match the supplied model files.");

    if(paths.find("metric") == paths.end() || paths.find("index_path") == paths.end())
        throw std::runtime_error("--metric and --index_path are required");
    if(norm_tolerance < 0.0)
        throw std::runtime_error("--norm_tolerance must be non-negative");

    const std::unordered_set<std::string> supported_modes = {
        "baseline", "twohop", "boundtwohop", "boundtwohopcmp", "history", "dabs"
    };
    if(!supported_modes.count(search_mode))
        throw std::runtime_error("Unsupported --search_mode: " + search_mode);

    Metric metric;
    if(paths["metric"]=="ip_float")
        metric=IP_float;
    else if(paths["metric"]=="l2_float")
        metric=L2_float;
    else
        throw std::runtime_error("Unsupported metric");

    auto hnsw_ngfix = new HNSW_NGFix<float>(metric, paths["index_path"]);
    hnsw_ngfix->bound_twohop_gate = jump_gate;
    if(jump_policy.usesTree()) jump_policy.loadTree(jump_tree_path, k);
    if(jump_policy.usesHistory()) jump_policy.loadHistory(jump_history_path, hnsw_ngfix->n.load(), k);
    jump_policy.validate();
    hnsw_ngfix->jump_policy = std::move(jump_policy);
    hnsw_ngfix->printGraphInfo();

    std::string result_base;
    if(!audit_prefix.empty()) result_base = StripCsvExtension(audit_prefix);
    else if(paths.find("result_path") != paths.end() && !paths["result_path"].empty())
        result_base = StripCsvExtension(paths["result_path"]);
    else
        result_base = "graph_audit";

    if(write_graph_audit || graph_audit_only) {
        {
            std::ofstream graph_eh_out(result_base + "_graph_eh_distribution.csv");
            hnsw_ngfix->writeGraphEhDistribution(graph_eh_out);
        }
        {
            std::ofstream out(result_base + "_repair_degree_distribution.csv");
            hnsw_ngfix->writeRepairDegreeDistribution(out);
        }
        {
            std::ofstream out(result_base + "_repair_degree_summary.csv");
            hnsw_ngfix->writeRepairDegreeSummary(out);
        }
        {
            std::ofstream out(result_base + "_repair_eh_by_saturation.csv");
            hnsw_ngfix->writeRepairEhBySaturation(out);
        }
    }

    if(graph_audit_only) {
        std::cout << "Graph audit complete. Prefix: " << result_base << std::endl;
        delete hnsw_ngfix;
        return 0;
    }

    std::cout<<"Search mode: "<<search_mode<<std::endl;
    std::cout << "Jump gate: cur_topk=" << jump_gate.cur_topk
              << " min_hops=" << jump_gate.min_hops << " cur_margin=" << jump_gate.cur_margin
              << " mid_gap=" << jump_gate.mid_gap << " max_mids_per_cur=" << jump_gate.max_mids_per_cur
              << " detour_ratio=" << jump_gate.detour_ratio
              << " max_nn_per_query=" << jump_gate.max_nn_per_query
              << " first_edge_filter=" << static_cast<int>(jump_gate.first_edge)
              << " second_edge_filter=" << static_cast<int>(jump_gate.second_edge)
              << " schedule=" << static_cast<int>(jump_gate.schedule)
              << " selection=" << static_cast<int>(jump_gate.selection)
              << " trigger=" << static_cast<int>(jump_gate.trigger)
              << " stall_window=" << jump_gate.stall_window
              << " max_nn_per_mid=" << jump_gate.max_nn_per_mid
              << " max_nn_per_cur=" << jump_gate.max_nn_per_cur << std::endl;
    const auto& policy = hnsw_ngfix->jump_policy;
    std::cout << "Jump policy: mode=" << static_cast<int>(policy.mode)
              << " tree=" << jump_tree_path << " threshold=" << policy.tree_threshold
              << " history=" << jump_history_path << " threshold=" << policy.history_threshold
              << " min_queries=" << policy.history_min_queries
              << " unknown_allow=" << policy.history_unknown_allow
              << " query_offset=" << query_offset << std::endl;

    if(paths.find("test_query_path") == paths.end() ||
       paths.find("test_gt_path") == paths.end() ||
       paths.find("result_path") == paths.end())
        throw std::runtime_error("search mode requires --test_query_path, --test_gt_path, and --result_path");

    size_t query_number = 0;
    size_t gt_number = 0;
    size_t test_gt_dim = 0;
    size_t vecdim = 0;

    auto test_query = LoadData<float>(paths["test_query_path"], query_number, vecdim);
    auto test_gt = LoadData<int>(paths["test_gt_path"], gt_number, test_gt_dim);
    if(query_number != gt_number)
        throw std::runtime_error("Query and ground-truth row counts differ.");
    if(vecdim != hnsw_ngfix->dim)
        throw std::runtime_error("Query dimension differs from the loaded index dimension.");
    if(k <= 0 || static_cast<size_t>(k) > test_gt_dim)
        throw std::runtime_error("K must be positive and no larger than the GT width.");
    if(query_offset >= query_number) throw std::runtime_error("--query_offset outside query file");
    const size_t test_number = std::min(query_number - query_offset, query_limit);
    if(test_number == 0)
        throw std::runtime_error("No queries selected; --query_limit must be positive.");

    const auto dabs_transform = metric == IP_float
        ? HNSW_NGFix<float>::DABSDistanceTransform::NORMALIZED_IP
        : HNSW_NGFix<float>::DABSDistanceTransform::SQUARED_L2;

    const bool must_audit_norms = (search_mode == "dabs" && metric == IP_float);
    if(check_norms || must_audit_norms) {
        const size_t indexed_vectors = hnsw_ngfix->n.load();
        const NormSummary base_norms = SummarizeNorms(
            indexed_vectors, norm_sample_size, vecdim, norm_tolerance,
            [hnsw_ngfix](size_t i) { return hnsw_ngfix->getData(static_cast<id_t>(i)); });
        const NormSummary query_norms = SummarizeNorms(
            query_number, norm_sample_size, vecdim, norm_tolerance,
            [test_query, vecdim](size_t i) { return test_query + i * vecdim; });

        std::ofstream norm_out(result_base + "_norm_audit.csv");
        norm_out << std::fixed << std::setprecision(9);
        norm_out << "source,population,sampled,mean_norm,min_norm,max_norm,"
                 << "outside_tolerance,nonfinite,tolerance,unit_normalized\n";
        WriteNormSummary(norm_out, "index_base", base_norms, norm_tolerance);
        WriteNormSummary(norm_out, "query", query_norms, norm_tolerance);

        std::cout << std::fixed << std::setprecision(9)
                  << "Norm audit: base mean/min/max=" << base_norms.meanNorm()
                  << "/" << base_norms.min_norm << "/" << base_norms.max_norm
                  << ", outside=" << base_norms.outside_tolerance
                  << "; query mean/min/max=" << query_norms.meanNorm()
                  << "/" << query_norms.min_norm << "/" << query_norms.max_norm
                  << ", outside=" << query_norms.outside_tolerance << std::endl;

        const bool unit_normalized =
            base_norms.unitNormalized() && query_norms.unitNormalized();
        if(must_audit_norms && !unit_normalized && !allow_nonunit_dabs) {
            throw std::runtime_error(
                "DABS with ip_float requires unit-normalized base/query vectors. "
                "See the generated _norm_audit.csv, or use --allow_nonunit_dabs 1 "
                "only for a clearly labeled heuristic run without the metric guarantee.");
        }
        if(must_audit_norms && !unit_normalized && allow_nonunit_dabs) {
            std::cerr << "WARNING: running DABS on non-unit IP vectors; "
                      << "the metric-distance interpretation and the DABS theorem's "
                      << "metric precondition do not apply.\n";
        }
    }

    if(search_mode=="history")
    {
        hnsw_ngfix->LoadHistoryTable(history_path);
        hnsw_ngfix->history_threshold = history_threshold;
        hnsw_ngfix->enable_history_twohop=true;
    }

    std::ofstream output;
    output.open(paths["result_path"]);
    if(!output) throw std::runtime_error("Cannot open result_path");

    std::ofstream eh_aggregate_out(result_base + "_search_eh_distribution.csv");

    std::ofstream nn_out;
    std::ofstream search_state_out;
    std::ostream* search_state_ptr = nullptr;
    std::ostream* nn_ptr = nullptr;
    std::ofstream jump_event_out;
    std::ofstream jump_mid_out;
    std::ofstream jump_hopbin_out;
    std::ostream* jump_event_ptr = nullptr;
    std::ostream* jump_mid_ptr = nullptr;
    std::ostream* jump_hopbin_ptr = nullptr;
    if(event_trace_ef >= 0) {
        if(trace_search_state) {
            search_state_out.open(result_base + "_probe_state.csv");
            if(!search_state_out) throw std::runtime_error("Cannot open search state trace");
            search_state_ptr=&search_state_out;
        }
        if(search_mode != "boundtwohopcmp")
            throw std::runtime_error("--event_trace_ef currently requires --search_mode boundtwohopcmp");
        if(nn_trace) {
            nn_out.open(result_base + "_nn_candidates.csv");
            if(!nn_out) throw std::runtime_error("Cannot open NN trace");
            nn_ptr = &nn_out;
        }
        jump_mid_out.open(result_base + "_jump_mids.csv");
        if(!jump_mid_out) throw std::runtime_error("Cannot open mid trace");
        if(!event_trace_compact) {
            jump_event_out.open(result_base + "_jump_events.csv");
            jump_hopbin_out.open(result_base + "_jump_hopbins.csv");
            if(!jump_event_out || !jump_hopbin_out) throw std::runtime_error("Cannot open event trace");
            jump_event_ptr = &jump_event_out;
            jump_hopbin_ptr = &jump_hopbin_out;
        }
        jump_mid_ptr = &jump_mid_out;
        std::cout << "Event trace enabled at ef=" << event_trace_ef << std::endl;
    }

    TestQueries<float>(
        output,
        test_query + query_offset * vecdim,
        test_gt + query_offset * test_gt_dim,
        test_number,
        k,
        test_gt_dim,
        vecdim,
        hnsw_ngfix,
        search_mode,
        ef_values,
        dabs_gammas,
        dabs_transform,
        &eh_aggregate_out,
        event_trace_ef,
        event_trace_query_limit,
        jump_event_ptr,
        jump_mid_ptr,
        jump_hopbin_ptr,
        query_offset,
        event_trace_compact, nn_ptr, search_state_ptr);

    delete []test_query;
    delete []test_gt;
    delete hnsw_ngfix;
    return 0;
}
