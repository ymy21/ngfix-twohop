"""Code-linked definitions for every observed CSV field in this experiment."""
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL='test/tools/result_evaluation.h'
GRAPH='ngfixlib/graph/hnsw_ngfix.h'
SNAP='ngfixlib/graph/bound_search_snapshot.h'
BENCH='test/bench_hnsw_unbounded_clean.cc'
FEATURE='analysis/analyze_bound_features.py'
SUMMARY='analysis/summarize_hnsw_mechanism.py'


def location(file,needle):
    for n,line in enumerate((ROOT/file).read_text(encoding='utf-8').splitlines(),1):
        if needle in line:return f'{file}:{n}'
    return file


COMMON={
 'ndc_ratio':('配对成本比','同图同 ef 二跳平均 NDC/paired beam 平均 NDC。不是相同 recall 的 NDC 比。','倍','analysis/build_complete_experiment_report.py',"s['ndc_ratio']"),
 'ndc_nn_share':('成本占比','mean(ndc_nn)/mean(ndc)，含入口的总成本作分母。','比例','analysis/write_complete_experiment_report.py',"ndc_nn_share"),
 'nearest_useful_probe_rate':('最近归因率','SUCCESS_RECOVERED_GT 的 probe 数/全部 probe 数；是最近来源标签，不是因果必要性。','比例','analysis/write_complete_experiment_report.py','nearest_useful_probe_rate'),
 'nearest_useful_nn_cost_share':('最近归因成本','最近 useful probe 的 fresh NN 成本/全部 fresh NN 成本。','比例','analysis/write_complete_experiment_report.py','nearest_useful_nn_cost_share'),
 'ancestor_only_nn_cost_share':('祖先成本','最近来源无新增、但祖先链有新增的 probe NN 成本/全部 NN 成本。','比例','analysis/write_complete_experiment_report.py','ancestor_only_nn_cost_share'),
 'shared_cur_recovery_share':('路径来源占比','baseline 扩展 cur、评分但未扩展 mid 的 probe 所归因恢复 GT/全部归因恢复 GT。','比例','analysis/write_complete_experiment_report.py','shared_cur_recovery_share'),
 'anchor_ef':('固定困难组','用于选择最低 recall 20% 查询的 HNSW beam ef，本轮为 200；选择后跨方法和 ef 固定查询 ID。','ef','analysis/compare_graph_k_matrix.py','anchor=200'),
 'mean_delta_ndc':('配对成本','固定查询组内 mean(NDC_twohop−NDC_beam)，并非相同 recall 的成本差。','完整距离/查询','analysis/analyze_hnsw_stage1.py','mean_delta_ndc='),
 'query_id':('标识','查询文件中的零基行号；本轮为 0–999。不是独立样本次数。','ID',EVAL,'i + query_offset'),
 'ef':('配置','ef 结果堆容量；本轮所有配置 ef≥K。未扩展候选队列可大于 ef。','节点',GRAPH,'Search_QuadHeap q0'),
 'k':('配置','最终返回近邻数，10 或 100；决定 GT 定义和评价，不等于 ef。','节点',BENCH,'size_t k=100'),
 'method':('配置','beam 或 bound_unbounded；相同图和 ef 下比较。','类别',BENCH,'bound_unbounded'),
 'mode':('配置','诊断入口使用 boundtwohopcmp，先运行 paired beam，再运行二跳。','类别',EVAL,'if(mode == "boundtwohopcmp")'),
 'row_type':('记录类型','QUERY 是单查询，SUMMARY 是汇总；分布统计排除 SUMMARY，避免双计。','类别',EVAL,'QUERY,'),
 'query_offset':('配置','查询起始行号，本轮为 0。','ID',BENCH,'--query-offset'),
 'queries':('样本数','依表语境为该测试点查询数，或该类别/特征桶中出现记录的不同 query_id 数。类别之间可重叠，不可直接相加。','查询',BENCH,'queries.count'),
 'round':('配置','计时轮次；本轮只有 round=0，没有轮次间方差。','轮',BENCH,'for(size_t round'),
 'rounds':('配置','重复计时轮数；本轮为 1。','轮',BENCH,'for(size_t round'),
 'verified_queries':('核对','每个测试点与参考搜索核对完整返回结果和 NDC 的查询数，min(verify_limit,query_count)=200。','查询',BENCH,'const size_t verified'),
 'batch_queries':('计时','original 模式等于测试查询数；batch 模式等于完整查询遍数×查询数。','查询',BENCH,'const uint64_t batch_queries'),
 'seconds':('计时','original 是 sum(latency_us)/1e6；batch 是包围整批搜索的 steady_clock 耗时。','秒',BENCH,'seconds=double(total_us)'),
 'qps':('计时','性能表是 batch_queries/seconds；诊断 QUERY 行是 1e6/该查询延迟，SUMMARY 行是 N×1e6/sum(latency_us)。不能平均逐查询 QPS。诊断包含统计开销，不能当作主性能。','查询/秒',EVAL,'result.qps ='),
 'latency_us':('计时','搜索调用前后 gettimeofday 差，微秒；诊断只包含二跳调用，paired baseline 被排除；诊断调用内仍有仪表开销。','微秒',EVAL,'result.latency_us ='),
 'checksum':('一致性','每次结果的 first_id+last_id×0x9e3779b97f4a7c15+size 的 uint64 累加；只是简短 fingerprint，不是完整结果的密码学哈希。','uint64',BENCH,'static uint64_t fingerprint'),
 'timing_protocol':('计时','original 逐查询计时/计时内 NDC，或 batch 整批计时/计时外 NDC；本轮为 original。','类别',BENCH,'timing_protocol="batch"'),
 'recall':('精度','|返回 top-K ∩ 精确 GT top-K|/K；性能汇总是等权查询平均。','比例',EVAL,'result.recall ='),
 'rderr':('精度口径','源码实际为 sum(d(ANN_i,q)/d(GT_i,q))/K，未减 1；|d(GT_i,q)|≤1e-5 的项跳过但分母仍为 K。精确结果通常接近 1，不是论文定义接近 0 的误差。','距离比',EVAL,'result.rderr ='),
 'ndc':('计算成本','搜索中完整 query-to-node 距离次数，含入口；ndc=ndc_entry+ndc_mid+ndc_nn。离线结构、GT/距离比计算不计入。','完整距离',EVAL,'result.ndc ='),
 'mean_ndc':('计算成本','计时外对所有查询 NDC 等权平均；与 original 计时轮的计数再核对。','完整距离/查询',BENCH,'point.mean_ndc='),
 'ndc_entry':('计算成本','入口完整 query 距离，一般为 1。原公开 beam 计数漏此项，新性能入口在计时后归一化补 1。','完整距离',EVAL,'result.ndc_entry ='),
 'ndc_mid':('计算成本','第一跳新节点的完整 query 距离次数；此前第二跳评分过的节点不会再计第一跳。下降可能是计算位移。','完整距离',EVAL,'result.ndc_mid ='),
 'ndc_nn':('计算成本','第二跳新节点的完整 query 距离次数；ndc_nn=ndc_nn_jump+ndc_nn_nojump。','完整距离',EVAL,'result.ndc_nn ='),
 'ndc_nn_jump':('计算成本','最终至少接纳一个 nn 的 probe，其全部新 NN 距离次数；并非只有被接纳 nn 的距离，更不表示最终答案收益。','完整距离',GRAPH,'stats.nn_jump_distance_cmps +='),
 'ndc_nn_nojump':('计算成本','最终一个 nn 都未接纳的 probe，其全部新 NN 距离次数。','完整距离',GRAPH,'stats.nn_nojump_distance_cmps +='),
 'hops':('搜索进展','实际扩展 cur 的次数；不计弹出但因边界停止的节点。','扩展',GRAPH,'++stats.cur_pop_count'),
 'total_hops':('事后进展','该查询最终扩展次数，等于 hops；用于归一化阶段，探测开始时未知。','扩展',EVAL,'stats.cur_pop_count'),
 'hop':('搜索进展','当前 cur 的一基扩展步号。','扩展',GRAPH,'++hop;'),
 'hop_frac':('事后进展','hop/最终 total_hops；不可直接作为在线触发输入。','比例',EVAL,'static_cast<double>(ev.hop)'),
 'hop_over_ef':('在线进展','max(hop−1,0)/ef；不是最终完成比例，可大于 1。','比例',EVAL,'hop_over_ef'),
 'mid_examined':('扫描成本','第一跳扫描的邻接项数，包含已访问目标和重复边。','邻接项',GRAPH,'++stats.onehop_mid_examined'),
 'twohop_mid_scans':('探测次数','实际扫描 N(mid) 的 probe 次数；等于 jump_mids+failed_jump_mids。','probe',GRAPH,'++stats.twohop_mid_scan_count'),
 'nn_checked':('扫描成本','第二跳扫描邻接项数，包括 visited/重复目标；full selector 下 ≥ndc_nn。','邻接项',GRAPH,'++stats.checked_nn_count'),
 'direct_mids':('接纳','bound 模式中被实时 beam 接纳的第一跳节点数；不是 mid_dist≤cur_dist 的旧 twohop 定义。','节点',GRAPH,'if(mid_admitted)'),
 'jump_mids':('接纳','至少接纳一个二跳候选的 probe 数，不等于改善 recall 的 probe 数。','probe',GRAPH,'++stats.jump_mid_count'),
 'failed_jump_mids':('接纳','接纳二跳候选数为 0 的 probe 数，包括无 fresh NN 的情况。','probe',GRAPH,'++stats.failed_jump_mid_count'),
 'skipped_mids':('visited','第一跳因 visited 跳过的邻接项次数；包含重复访问同一目标。','邻接项',GRAPH,'++stats.skipped_mid_count'),
 'jump_candidates':('接纳','通过二跳满足实时 beam 条件并入队的节点数；本查询同一 fresh 节点只评分一次。','节点',GRAPH,'++stats.jump_candidate_count'),
 'jump_candidates_gt':('事后标签','被二跳接纳的节点中属于当前 K 的 GT 的数量，包括 baseline 已命中的 GT。','节点',EVAL,'++result.jump_candidates_gt'),
 'paired_baseline_recall':('配对对照','同一 q、同图、同 ef、同 K 的 beam 最终 recall；不是另一张图，也不是较大 ef 的 beam。','比例',EVAL,'result.paired_baseline_recall ='),
 'paired_baseline_ndc':('配对对照','同一配置 beam 的搜索 NDC，包含入口。','完整距离',EVAL,'result.paired_baseline_ndc ='),
 'recovered_gt_total':('事后收益','二跳命中 GT 集合减去 beam 命中集合的大小；T 为二跳结果，B 为同图同 ef beam 结果。','GT',EVAL,'++result.recovered_gt_total'),
 'lost_gt_total':('事后损失','beam 命中 GT 集合减去二跳命中集合的大小；净 recall 差=(recovered−lost)/K。','GT',EVAL,'++result.lost_gt_total'),
 'recovered_gt_direct_jump':('事后归因','recovered GT 本身属于 jump_admitted_nodes；优先归为直接，不再计路由。','GT',EVAL,'++result.recovered_gt_direct_jump'),
 'recovered_gt_bridge_descendant':('事后归因','非直接的 recovered GT 有最近 jump root 来源；不等于严格因果收益。','GT',EVAL,'++result.recovered_gt_bridge_descendant'),
 'recovered_gt_non_gt_bridge_root':('事后归因','路由 recovered GT 中，最近 jump root 不是当前 K 的 GT 的部分；是路由的子集，不能再相加。','GT',EVAL,'++result.recovered_gt_non_gt_bridge_root'),
 'final_gt_direct_jump':('最终归因','最终命中 GT 且本身经 jump 入队的数量，包含 baseline 共享 GT。','GT',EVAL,'++result.final_gt_direct_jump'),
 'final_gt_bridge_descendant':('最终归因','最终命中 GT 的非直接 jump 后继数量，包含 baseline 共享 GT。','GT',EVAL,'++result.final_gt_bridge_descendant'),
 'final_gt_non_gt_bridge_root':('最终归因','最终 GT 最近 root 非 GT 的数量；与直接/路由分类重叠，是子集视图。','GT',EVAL,'++result.final_gt_non_gt_bridge_root'),
 'final_topk_bridge_descendant':('最终归因','最终 top-K 中有 jump root 来源的节点数，包含 root 本身及非 GT。','节点',EVAL,'++result.final_topk_bridge_descendant'),
 'checked_nn':('单 probe 成本','该 mid 探测扫描的邻接项数量，含已访问/重复目标。','邻接项',GRAPH,'mev.checked_nn ='),
 'fresh_nn_cmps':('单 probe 成本','该 mid 探测新计算的完整 query-to-nn 距离数。','完整距离',GRAPH,'mev.fresh_nn_cmps ='),
 'admitted_nn':('单 probe 接纳','该 mid 探测最终通过实时 beam 条件的 nn 数量。','节点',GRAPH,'mev.admitted_nn ='),
 'success':('单 probe 标签','admitted_nn>0；与最终新增 GT 不同。','0/1',EVAL,'const bool success ='),
 'useful_mid':('单 probe 标签','direct_new_gt>0 或 nearest_recovered_desc_gt>0；最近来源归因，可能漏掉仅作为祖先帮助后续 jump 的 probe。','0/1',EVAL,'const bool useful_mid ='),
 'direct_new_gt':('单 probe/分箱归因','该 mid 的 jump root 本身为最终新增 GT 的数量。','GT',EVAL,'++mo.direct_new_gt'),
 'direct_shared_gt':('单 probe/分箱归因','该 mid 的直接 jump GT 也已被 baseline 返回；不算新增。','GT',EVAL,'++mo.direct_shared_gt'),
 'sanity_gt_not_final':('健全性','接纳了当前 K 的 GT 但其未留在最终二跳 top-K 的事件数；不能忽略，可能与同距离 GT tie 有关。','事件',EVAL,'++mo.sanity_gt_not_final'),
 'non_gt_roots':('单 probe 标签','接纳 nn 不属于当前 K 的 GT 的数量；K=10 与 K=100 的非 GT 定义不同。','root',EVAL,'++mo.non_gt_roots'),
 'useful_roots_nearest':('单 probe 标签','直接新增 GT 或最近来源 recovered 后继>0 的 root 数。','root',EVAL,'++mo.useful_roots_nearest'),
 'useful_roots_ancestor':('祖先归因','直接新增 GT 或祖先链 recovered 后继>0 的 root 数；包含嵌套 jump 的上游根。','root',EVAL,'++mo.useful_roots_ancestor'),
 'mid_checked_nn':('单 jump 上下文','该 jump 所属 probe 的总邻接项扫描数；同 probe 多 jump 行会重复，不能按事件直接求和算总成本。','邻接项',GRAPH,'ev.mid_checked_nn ='),
 'mid_fresh_nn_cmps':('单 jump 上下文','该 jump 所属 probe 的全部新 NN 距离数；同 probe 多事件重复，不能直接求和。','完整距离',GRAPH,'ev.mid_fresh_nn_cmps ='),
 'nn_gt_rank':('事后标签','nn 在当前精确 top-K 中的一基排名；非 GT 为 −1。','排名',EVAL,'const int nn_gt_rank ='),
 'rank_in_fresh_nn_by_query_dist':('事后候选排序','所属 probe 的所有 fresh NN 按完整 query 距离排序时的排名；已付距离成本，不是探测前可得的筛选信号。','排名',GRAPH,'ev.rank_in_fresh_nn_by_query_dist ='),
 'probe_id':('标识','本查询中一基实际 probe 序号；跨 query 重置。需与 query_id、ef 联用。','ID',GRAPH,'trace_probe_id='),
 'mid_gap':('在线距离状态','mid_dist−拒绝时 beam bound；因严格接纳规则通常≥0。','IP距离',EVAL,'const double mid_gap ='),
 'nn_margin':('接纳状态','接纳 nn 前的实时 bound−nn_dist；不是 cur_dist−nn_dist。','IP距离',EVAL,'const double nn_margin ='),
 'mid_gap_normalized':('在线派生','(mid_dist−拒绝时 bound)/max(|bound|,1e−6)，启发式尺度归一化，无三角不等式保证。','比例',FEATURE,"df['mid_gap_normalized']"),
 'cur_slack_normalized':('在线派生','(拒绝时 bound−cur_dist)/max(|bound|,1e−6)。','比例',FEATURE,"df['cur_slack_normalized']"),
 'nn_margin_normalized':('接纳后派生','(接纳前 bound−nn_dist)/max(|bound|,1e−6)，需要已评分 nn。','比例',FEATURE,"df['nn_margin_normalized']"),
 'topk_spread':('在线派生','topk_q75−topk_q25；当前 top-K 距离四分位距。','IP距离',FEATURE,"df['topk_spread']"),
 'jump_class':('事后类别','DIRECT_GT_NEW/SHARED、ROUTER_TO_GT/NO_GT 或健全性异常。ROUTER_TO_GT 按任意最终 GT 祖先后继判定，不保证新增 GT。','类别',EVAL,'std::string cls;'),
 'probe_class':('事后类别','失败无 fresh、失败有 fresh、成功无最近归因新增 GT、成功有最近归因新增 GT。不是因果可删除标签。','类别',FEATURE,"df['probe_class']"),
 'useful':('事后标签','单 probe 的 direct_new_gt+nearest_recovered_desc_gt>0。','0/1',FEATURE,"df['useful']"),
 'recovered_gt':('最近归因收益','单 probe 的 direct_new_gt+nearest_recovered_desc_gt；保留 loss 在查询级计算净收益。','GT',FEATURE,"df['recovered_gt']"),
 'feature':('特征标识','该行统计的实际变量名；均值/分位数等针对此变量及该类别。','名称',FEATURE,'for feature in features'),
 'group':('分组标识','probe_class 或 jump_class 名称；按 ef、图、K 分开。','类别',FEATURE,'def descriptions'),
 'feature_bin':('分桶','≤5 个唯一值时按值，否则 pd.qcut 四分位分桶并去重边界。稀疏变量可能退化为一桶，不代表恒定。','区间/值',FEATURE,"valid['feature_bin']"),
 'events':('记录量','该分组的事件行数；不是不同 query 数，也不总是不同全局节点数。','行',FEATURE,"events=len(g)"),
 'probes':('记录量','实际探测 mid 的行数；失败也包括。','probe',FEATURE,'probes=len(g)'),
 'success_rate':('接纳率','桶内 admitted_nn>0 的 probe 数/桶内 probe 数。','比例',FEATURE,'success_rate='),
 'useful_rate':('最近归因有效率','桶内 useful probe 数/桶内 probe 数；有祖先贡献但最近归因为零的 probe 会被排除。','比例',FEATURE,'useful_rate='),
 'query_balanced_useful_rate':('查询平衡有效率','先对桶中每个出现 probe 的 query 算 useful 比例，再对这些 query 等权平均；不同桶 query 集合可能不同。','比例',FEATURE,'query_balanced_useful_rate='),
 'fresh_nn_ndc':('分桶/阶段成本','所有属于该桶或阶段的 probe 的 fresh_nn_cmps 之和；不是事件行重复上下文的和。','完整距离',EVAL,'ba.fresh_nn_ndc +='),
 'recovered_gt_per_1k_fresh_nn_ndc':('归因产出','1000×该桶归因新增 GT/fresh_nn_ndc；没有扣查询级 lost，不是净 ROI 或反事实筛选效果。','GT/千距离',FEATURE,'recovered_gt_per_1k_fresh_nn_ndc='),
 'net_gt_per_1k_nn_ndc':('查询净产出','1000×sum(recovered_gt_total−lost_gt_total)/sum(ndc_nn)。','净GT/千距离',SUMMARY,'net_gt_per_1k_nn_ndc='),
 'no_admission_nn_cost_share':('成本占比','sum(ndc_nn_nojump)/sum(ndc_nn)。','比例',SUMMARY,'no_admission_nn_cost_share='),
 'duplicate_or_visited_scan_fraction':('扫描重复','(sum(nn_checked)−sum(ndc_nn))/sum(nn_checked)；包括 visited，不只重复边。','比例',SUMMARY,'duplicate_or_visited_scan_fraction='),
 'successful_probe_rate':('接纳率','sum(jump_mids)/sum(twohop_mid_scans)，是所有 probe 加权率。','比例',SUMMARY,'successful_probe_rate='),
 'admitted_per_fresh_nn':('接纳收益','sum(jump_candidates)/sum(ndc_nn)。','比例',SUMMARY,'admitted_per_fresh_nn='),
 'net_recovered_gt_mean':('查询净收益','mean(recovered_gt_total−lost_gt_total)。','GT/查询',SUMMARY,'net_recovered_gt_mean='),
 'delta_recall':('配对精度','mean(recall−paired_baseline_recall)。','比例',SUMMARY,'delta_recall='),
 'delta_ndc':('配对成本','mean(ndc−paired_baseline_ndc)。','完整距离/查询',SUMMARY,'delta_ndc='),
 'improved_fraction':('查询分布','recall_twohop>recall_beam 的查询比例；不按新增数加权。','比例',SUMMARY,'improved_fraction='),
 'worsened_fraction':('查询分布','recall_twohop<recall_beam 的查询比例。','比例',SUMMARY,'worsened_fraction='),
 'direct_recovered_share':('归因占比','sum(recovered_gt_direct_jump)/sum(recovered_gt_total)。','比例',SUMMARY,'direct_recovered_share='),
 'routing_recovered_share':('归因占比','sum(recovered_gt_bridge_descendant)/sum(recovered_gt_total)。','比例',SUMMARY,'routing_recovered_share='),
 'non_gt_router_recovered_share':('归因子集','sum(recovered_gt_non_gt_bridge_root)/sum(recovered_gt_total)，是路由的子集。','比例',SUMMARY,'non_gt_router_recovered_share='),
 'unattributed_recovered_share':('归因完整性','(recovered−direct−routing)/recovered，未归因部分；不能强行归入路由。','比例',SUMMARY,'unattributed_recovered_share='),
 'scans':('阶段机会','阶段 bin 内的实际 probe 数。','probe',EVAL,'++ba.scans'),
 'successful_mids':('阶段接纳','该 bin 内 admitted_nn>0 的 probe 数。','probe',EVAL,'++ba.successful_mids'),
 'failed_mids':('阶段接纳','该 bin 内 admitted_nn=0 的 probe 数。','probe',EVAL,'++ba.failed_mids'),
 'useful_mids':('阶段最近归因','该 bin 内 direct_new_gt>0 或 recovered_desc_gt_nearest>0 的 probe 数。','probe',EVAL,'++ba.useful_mids'),
 'bin':('事后阶段','min(19,floor(max(hop−1,0)×20/max(total_hops,1)))；总步数事后可知。','0–19',EVAL,'static constexpr size_t BINS'),
 'hop_frac_lo':('事后阶段','bin/20；闭区间下界。','比例',EVAL,'static_cast<double>(b) / BINS'),
 'hop_frac_hi':('事后阶段','(bin+1)/20；区间上界。','比例',EVAL,'static_cast<double>(b + 1) / BINS'),
 'mid_gap_sum':('阶段聚合','bin 内 sum(mid_dist−bound_at_reject)。','IP距离之和',EVAL,'ba.mid_gap_sum +='),
 'bound_sum':('阶段聚合','bin 内拒绝时边界之和。除以 scans 得平均 bound。','IP距离之和',EVAL,'ba.bound_sum +='),
 'mid_gap_mean':('阶段派生','mid_gap_sum/scans 或该类别事件的平均 mid_gap，依表语境。','IP距离',FEATURE,'descriptions'),
 'bound_recall':('精度','二跳最终 recall，与主查询表 recall 相同。','比例',EVAL,'result.recall'),
 'ratio':('比例','图 EH 表为该 EH 的 NGFix 边数/全部 NGFix 边数，排除 RFix；其它分布按表中总数归一。','比例',GRAPH,'writeGraphEhDistribution'),
 'count':('频数','具体表中所列 EH/类别的计数，不自动代表查询数。','计数',GRAPH,'writeGraphEhDistribution'),
 'eh':('边硬度','有限 EH 0–200 或 EH_INF；按该表标识；RFix 用单独类别，不当有限 EH。','EH/标记',GRAPH,'ehHistIndex'),
 'eh_group':('硬度分组','修复区域按有限 EH 范围、EH_INF 或 RFix 分组。','类别',GRAPH,'writeRepairEhBySaturation'),
 'stage':('边生命周期','examined 扫描、scored 新评分、admitted 入队、expanded 后续实际扩展；分母各不相同。','类别',EVAL,'emit_search_eh'),
 'ratio_within_stage_ngfix':('阶段比例','该 stage 中该 EH 的 NGFix 边使用次数/该 stage 全部 NGFix 边次数；排除 HNSW/RFix。','比例',EVAL,'ratio_within_stage_ngfix'),
 'repair_degree':('图结构','该节点最终修复区域中有效边条目数，含 NGFix 和 RFix。','有向边条目',GRAPH,'getRepairDegreeStats'),
 'node_count':('图结构','图节点数，或该 repair_degree 桶的节点数，依表语境。','节点',GRAPH,'writeRepairDegreeDistribution'),
 'ratio_all_nodes':('图分布','该修复度数桶节点数/全部节点数。','比例',GRAPH,'writeRepairDegreeDistribution'),
 'metric':('标识','图结构汇总中的指标名称；其 value 不能跨不同单位平均。','名称',GRAPH,'writeRepairDegreeSummary'),
 'value':('图结构值','该行 metric 的值，可能为节点数、边数、比例或分位数；按 metric 解释。','依metric',GRAPH,'writeRepairDegreeSummary'),
 'source':('规范检查','index_base 或 query；节点范数审计的数据来源。','类别','test/search_hnsw_ngfix.cc','WriteNormAudit'),
 'population':('范数审计','被审计文件的总向量数；query 为原文件 10k，不等于本轮测试 1k。','向量','test/search_hnsw_ngfix.cc','summary.population'),
 'sampled':('范数审计','实际计算范数的向量数量；基础数据仅抽样 10k。','向量','test/search_hnsw_ngfix.cc','summary.sampled'),
 'mean_norm':('范数审计','sampled 向量 L2 范数平均值。','L2范数','test/search_hnsw_ngfix.cc','summary.meanNorm'),
 'min_norm':('范数审计','审计样本最小 L2 范数。','L2范数','test/search_hnsw_ngfix.cc','summary.min_norm'),
 'max_norm':('范数审计','审计样本最大 L2 范数。','L2范数','test/search_hnsw_ngfix.cc','summary.max_norm'),
 'outside_tolerance':('范数审计','|norm−1|>tolerance 的样本数。','向量','test/search_hnsw_ngfix.cc','summary.outside_tolerance'),
 'nonfinite':('范数审计','含非有限数值的审计样本数。','向量','test/search_hnsw_ngfix.cc','summary.nonfinite'),
 'tolerance':('范数审计','接近单位范数的容差，本轮 0.001。','范数差','test/search_hnsw_ngfix.cc','tolerance'),
 'unit_normalized':('范数审计','审计样本均有限且都在容差内则为 1；不能据抽样证明全部 2.505M 数据严格单位范数。','0/1','test/search_hnsw_ngfix.cc','summary.unitNormalized'),
}

STATE={
 'step':'当前实际扩展步号，等于记录 hop。',
 'ndc':'探测前累计完整 query 距离，含刚评分的 mid 和入口；1+state_ndc_mid+state_ndc_nn。',
 'ndc_mid':'探测前累计第一跳完整 query 距离。',
 'ndc_nn':'探测前累计第二跳完整 query 距离，不含本次 probe。',
 'result_updates':'截至此时非删除节点进入 ef 结果堆的次数，含入口。',
 'topk_updates':'截至此时实际接纳节点使 top-K 填充或严格距离改善的次数，含入口；等距离 ID 替换不计。',
 'steps_since_topk_update':'当前 hop−最后一次 top-K 更新 hop；不是墙钟停滞时间。',
 'discovery_depth':'cur 的首次接纳路径深度：入口 0，普通接纳 +1，二跳接纳 +2；不是图最短路或层号。',
 'frontier_size':'未扩展 candidate_set 的当前大小，可能含随后不再满足 bound 的旧候选，可大于 ef。',
 'beam_size':'当前 ef 结果堆有效大小；触发拒绝时通常已满，因此本轮此列往往等于 ef。',
 'topk_size':'min(K,beam_size)。',
 'entry_dist':'该查询初始入口的 1−dot(q,x)；同一 query 恒定，事件平均会按 probe 数加权。',
}


def definition(field,scope=''):
    if field in COMMON:
        group,text,unit,file,needle=COMMON[field]
        return dict(category=group,definition=text,unit=unit,code=location(file,needle),availability='事后/配置' if group.startswith(('事后','最终','计时','归因','阶段','范数')) else '按定义区分')
    if field.startswith('state_') and field[6:] in STATE:
        return dict(category='探测前搜索状态',definition=STATE[field[6:]],unit='IP距离' if field.endswith('dist') else '计数',code=location(SNAP,'captureBoundSearchSnapshot'),availability='探测前可知；采集有成本')
    m=re.fullmatch(r'(beam|topk)_(min|max|mean|variance|median|q25|q75)',field)
    if m:
        setname='当前 ef 结果堆全部距离' if m[1]=='beam' else '当前 ef 结果堆中最近 min(K,size) 个距离'
        stat={'min':'最小值','max':'最大值','mean':'算术均值','variance':'总体方差 Σ(d−mean)²/n','median':'线性插值 50% 分位数','q25':'线性插值 25% 分位数','q75':'线性插值 75% 分位数'}[m[2]]
        return dict(category='探测前距离分布',definition=f'{setname}的{stat}；复制有效槽排序，不修改堆。',unit='IP距离²' if m[2]=='variance' else 'IP距离',code=location(SNAP,'summarizeBoundDistances'),availability='探测前可知；提取有成本')
    m=re.fullmatch(r'(cur|mid|nn|best_nn)_(total_degree|hnsw_degree|ngfix_degree|rfix_degree|repair_degree|repair_ratio|ngfix_finite_degree|ngfix_inf_degree)',field)
    if m:
        role,metric=m.groups();meaning={'total_degree':'全部有效出边条目数','hnsw_degree':'原始 HNSW 出边条目数','ngfix_degree':'NGFix 出边条目数','rfix_degree':'RFix 出边条目数','repair_degree':'NGFix+RFix 出边条目数','repair_ratio':'repair_degree/total_degree，零度时为 0','ngfix_finite_degree':'有限 EH 的 NGFix 出边条目数，排除 RFix','ngfix_inf_degree':'EH_INF 的 NGFix 出边条目数'}[metric]
        return dict(category='局部图结构',definition=f'{role} 节点的{meaning}；边条目度数不去重。best_nn 仅在存在 fresh 候选时有效。',unit='比例' if metric=='repair_ratio' else '边条目',code=location(GRAPH,'getNodeLocalStructure'),availability='静态可提取；在线/缓存有成本')
    m=re.fullmatch(r'(cur_mid|mid_nn|cur_nn|mid_best_nn|cur_best_nn)_(data_dist|common_neighbors|union_neighbors|jaccard)',field)
    if m:
        pair,metric=m.groups();meaning={'data_dist':'1−dot(x_a,x_b)，不计搜索 query NDC','common_neighbors':'去重的有向出邻居集合交集大小','union_neighbors':'去重的有向出邻居集合并集大小','jaccard':'交集大小/并集大小；并集为空为 0'}[metric]
        return dict(category='局部拓扑/几何',definition=f'{pair} 两节点的{meaning}；不是无向社区标签或欧式距离。无 best_nn 时不适用。',unit='比例' if metric=='jaccard' else 'IP距离' if metric=='data_dist' else '邻居',code=location(GRAPH,'getPairLocalStructure'),availability='静态可提取；提取有成本')
    if field.endswith('_edge_type') or field in ['first_edge_type','second_edge_type']:
        text='实际遍历的 cur→mid 边' if field=='first_edge_type' else '实际遍历的 mid→nn 边' if field=='second_edge_type' else field.replace('_edge_type','').replace('_to_','→')+' 方向边'
        return dict(category='边类型',definition=text+'类型：0=HNSW，1=NGFix，2=RFix；−1=无该边/无候选。直接查询两节点时返回首个匹配边类型，多类型平行边不展开为集合。',unit='类别',code=location(GRAPH,'getSearchEdgeType'),availability='静态/遍历时可知')
    if field.endswith('_eh_raw') or field=='best_scored_second_edge_eh':
        return dict(category='边硬度标记',definition='实际边原始 EH：65535=EH_INF（base 边也用此占位），201=RFix 标记，有限 NGFix EH 为 0–200；不是连续难度 65535。无 best NN 时 −1。',unit='标记',code=location(GRAPH,'getSearchEdgeEh'),availability='边属性可知；不是 GT 标签')
    m=re.fullmatch(r'(exam|score|jump)_pair_([HNR]{2})',field)
    if m:
        stage,pair=m.groups();description={'exam':'扫描第二跳邻接项，含 visited/重复','score':'第二跳 fresh 目标实际算完整 query 距离','jump':'第二跳目标满足实时 beam 条件入队'}[stage]
        return dict(category='二跳边对机会/收益',definition=f'{pair}（H=原始，N=NGFix，R=RFix）路径的{description}次数；score 分母和 exam 不同。',unit='路径/目标次数',code=location(GRAPH,'recordEdgePair'),availability='搜索过程中累计')
    m=re.fullmatch(r'(examined|scored|admitted|expanded)_(total|hnsw|ngfix|rfix|ngfix_finite|ngfix_inf)',field)
    if m:
        stage,kind=m.groups();meaning={'examined':'邻接项扫描（可重复）','scored':'fresh 节点完整 query 评分','admitted':'通过实时 beam 入队','expanded':'以后实际扩展，并归于首次发现的入边'}[stage]
        return dict(category='边生命周期',definition=f'第一跳+第二跳中 {kind} 类型的{meaning}次数；不含入口。expanded 不含入口扩展，scored_total+1=ndc。',unit='次数',code=location(EVAL,'const auto& f ='),availability='搜索累计')
    m=re.fullmatch(r'(exam|score|admit|expand)_eh_(.+)',field)
    if m:
        stage,band=m.groups()
        return dict(category='NGFix EH 生命周期',definition=f'{stage} 阶段两跳合计、有限 EH 区间 {band} 的 NGFix 边次数；inf 为 EH_INF；排除 HNSW 与 RFix。',unit='次数',code=location(EVAL,'QUERY_EH_BUCKET_NAMES'),availability='搜索累计')
    if field.startswith('edges_'):
        return dict(category='邻接扫描',definition=f'{field} 是本查询两跳合计的邻接扫描次数，不是全图边数；与对应 examined 列重复。',unit='邻接项',code=location(EVAL,'result.edges_total ='),availability='搜索累计')
    if field.startswith('skipped_mid_from_'):
        return dict(category='距离位移诊断',definition='唯一第一跳目标此前已在成功/失败二跳 probe 评分、或已被 jump 接纳，导致本次被 visited 跳过。admitted_jump 是成功 probe 的子集，几列不能相加当总跳过。',unit='唯一目标/查询',code=location(GRAPH,'attributed_success_probe_skips'),availability='事后来源诊断')
    if field.startswith(('nearest_','ancestor_')):
        inclusive=field.startswith('ancestor_');recovered='recovered' in field;shared='shared' in field
        return dict(category='后继归因',definition=('祖先链包含式' if inclusive else '最近 jump root 唯一')+'后继 GT 数；'+('新增 GT' if recovered else 'baseline 共享 GT' if shared else '所有最终 GT')+'。root 自己不是自己的后继。'+('不同祖先可能重叠，不能求和当唯一新增。' if inclusive else '用于不重复的后继来源统计。'),unit='GT',code=location(EVAL,'root_recovered_gt_desc_nearest'),availability='需要最终结果和 GT')
    if field=='recovered_desc_gt':return definition('nearest_recovered_desc_gt',scope)
    if field in ['cur_id','mid_id','nn_id','best_scored_nn','best_admitted_nn']:
        return dict(category='节点标识',definition=field+' 节点 ID；best_scored 为该 probe 中完整 query 距离最小的 fresh NN，best_admitted 为最优被接纳 NN；无候选时 −1。ID 数值不是空间坐标或社区。',unit='ID',code=location(EVAL,'ev.best_scored_nn'),availability='按角色区分，best_* 是评分后')
    if field in ['cur_dist','mid_dist','nn_dist','best_scored_nn_dist','best_admitted_nn_dist']:
        return dict(category='查询几何',definition='query 与对应节点的 1−dot 距离，越小越近；nn/best_* 需要第二跳评分后；无候选为 NaN。',unit='IP距离',code=location(GRAPH,'getQueryDist'),availability='cur/mid 探测前，nn/best_* 探测后')
    if field in ['bound_at_reject','bound_at_mid_reject','bound_before_nn_admit']:
        return dict(category='beam 状态',definition='mid 被拒绝时，或当前 nn 被接纳之前的实时最差 ef 结果距离；两者可能因先前接纳而不同。',unit='IP距离',code=location(GRAPH,'bound_at_mid_reject ='),availability='实时搜索状态')
    if field.startswith('baseline_') or field.startswith('best_scored_baseline_'):
        return dict(category='paired beam 可见性',definition=field+'：同图同 ef 同 K 的 beam 是否评分/扩展/最终命中该节点。scored_not_expanded 不等同于已确认在相同状态被拒绝。',unit='0/1',code=location(EVAL,'paired_baseline_scored_nodes'),availability='需要额外 baseline 对照；不是单搜索输入')
    if field.endswith('_is_gt') or field in ['nn_is_gt','cur_is_gt','mid_is_gt']:
        return dict(category='真值标签',definition='对应节点是否属于当前查询的精确 top-K；K=10 时 rank 11–100 不属于 GT，K=100 时属于。',unit='0/1',code=location(EVAL,'gt_lookup.count'),availability='离线 GT 标签')
    if field in ['bound_final','bound_final_gt','best_scored_bound_final_gt']:
        return dict(category='最终标签',definition='该节点是否进入最终二跳 top-K；带 gt 时还要求属于精确 GT。',unit='0/1',code=location(EVAL,'bound_final_nodes'),availability='搜索结束后')
    if field=='dabs_gamma':
        return dict(category='未使用配置',definition='DABS 配置参数，本轮不用 DABS，因此 NaN 是不适用，不是缺失实验。',unit='不适用',code=location(EVAL,'quiet_NaN'),availability='配置')
    if field.startswith('gate_') or field=='jump_gate_active':
        return dict(category='门控诊断',definition='原 BoundGateStats 对应计数：拒绝、阻断、选择、触发或 cheap/shadow 工作量。本轮全部门控/预算关闭、immediate/full；除 rejected 和 probed 边类型等外许多列为 0。onehop/twohop_topk_changes 在 deferred 追踪关闭时为 0，不能据此判断没有进展。',unit='计数/标记',code=location('ngfixlib/graph/bound_twohop_gate.h','struct BoundGateStats'),availability='门控内部累计，部分本轮不启用')
    if field in ['first_h','first_n','first_r']:
        return dict(category='阶段边类型',definition='该阶段中第一跳类型为 H/N/R 的实际 probe 数。',unit='probe',code=location(EVAL,'++ba.first_h'),availability='事后阶段聚合')
    if field.endswith('_mean') and field[:-5] in COMMON:
        base=definition(field[:-5],scope);base['definition']='对所有查询等权平均：'+base['definition'];return base
    if field in ['mean','median','p25','p75','p90','min','max','std','p05','p95','p99']:
        return dict(category='分布汇总',definition='该行 feature/field 在指定图、K、ef、类别内的统计量；不是再跨不同特征平均。分位数线性插值，报告 std 为总体标准差；保留有效样本数。',unit='随feature',code=location(FEATURE,'def descriptions'),availability='聚合结果')
    if field in ['graph_label','graph','case','label','role','edge_pair','local_region','nn_visibility']:
        return dict(category='分组标识',definition='图/配置/节点角色/边对/局部 repair composition/paired beam 可见性的类别；local_region 不是社区划分。',unit='类别',code=location(FEATURE,"events['local_region']"),availability='分组派生')
    if field=='non_hh_examined' or field=='non_hh_admitted':
        return dict(category='实验污染检查',definition='所有非 HH 边对的扫描或接纳次数；Bottom 必须为 0。',unit='次数',code=location(SUMMARY,'non_hh_examined='),availability='累计检查')
    if field in ['examined','admitted','fresh_scored','admitted_per_fresh_scored']:
        return dict(category='边对汇总',definition='按 edge_pair 汇总扫描、fresh 评分或实际接纳；admitted_per_fresh_scored=admitted/fresh_scored，零分母留空。',unit='次数/比例',code=location(SUMMARY,'def edge_pair_summary'),availability='聚合')
    if field in ['recovered_descendants','queries_with_event']:
        return dict(category='事件汇总',definition='该类别的最近归因 recovered 后继 GT 总数，或出现此类事件的不同查询数。不同类别查询集合可重叠。',unit='GT/查询',code=location(SUMMARY,'def event_summary'),availability='事后聚合')
    if field in ['unique_nodes','top_1pct_nodes_event_share']:
        return dict(category='节点复用',definition='按图/K/ef/jump_class/role 统计不同节点数；将节点事件频数降序，前 ceil(unique_nodes×1%) 的事件数/全部事件数。',unit='节点/比例',code=location(FEATURE,'reuse=[]'),availability='全量事件聚合')
    if field.startswith(('qps_','ndc_p','latency_')) or field=='equivalent_mean_query_us':
        return dict(category='性能分布',definition='QPS 后缀为计时轮分布（本轮单轮 min/median/max 相同）；ndc_pXX 为查询 NDC 分位数；latency 后缀为逐查询微秒分布；equivalent_mean_query_us=1e6/qps_median。',unit='按字段为QPS/完整距离/微秒',code=location('analysis/analyze_hnsw_stage1.py','def summarize'),availability='性能汇总')
    if any(field.startswith(prefix) for prefix in ['beam_','bound_unbounded_','hnsw_','ngfix_','cheap_']) or field in ['target_recall','needs_denser_ef_grid','recall_delta_pp','attained_recall_span_pp','bound_minus_beam_recall_pp','bound_over_beam_qps','bound_over_beam_ndc']:
        return dict(category='性能目标匹配',definition='达到 target_recall 的最快实测点及其 ef/实际 recall/QPS/NDC；没有达到时为空，不外推。速度比按字段两方法相除；pp 为百分比点。needs_denser_ef_grid 比较实际 recall 差是否>0.001。',unit='按后缀',code=location('analysis/compare_graph_k_matrix.py','def target_table'),availability='实测曲线匹配')
    if field in ['total_edges','from_source_degree_lt_MEX','from_source_degree_eq_MEX']:
        return dict(category='最终图饱和代理',definition='该 EH 类修复边总数，及最终 source 修复度数小于/等于 MEX 的边数；不是插边当时的满度状态。',unit='有向修复边',code=location(GRAPH,'writeRepairEhBySaturation'),availability='最终图静态审计')
    return None
