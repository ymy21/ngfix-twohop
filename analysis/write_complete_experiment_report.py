"""Create the Chinese narrative, searchable HTML appendices and workbook data."""
from pathlib import Path
import base64
import html
import json
import re
import numpy as np
import pandas as pd
from build_complete_experiment_report import ROOT,PERF,MECH,OUT,TABLES,save_json,save_table
from report_field_catalog import definition,location,GRAPH

ORDER=['hnsw_k10_ef20','hnsw_k10_ef100','hnsw_k100_ef100','hnsw_k100_ef400',
       'ngfix_k10_ef20','ngfix_k10_ef100','ngfix_k100_ef100','ngfix_k100_ef400']


def read(name):return pd.read_csv(TABLES/(name+'.csv'))


def fmt(value):
    if value is None or pd.isna(value):return '不适用'
    if isinstance(value,(bool,np.bool_)):return '是' if value else '否'
    if isinstance(value,(int,np.integer)):return f'{value:,}'
    if isinstance(value,(float,np.floating)):
        if value and abs(value)<.0001:return f'{value:.5e}'
        if abs(value)>=1000:return f'{value:,.3f}'.rstrip('0').rstrip('.')
        return f'{value:.6f}'.rstrip('0').rstrip('.')
    return str(value)


def md_table(frame):
    headers=[str(c) for c in frame.columns]
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for values in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(fmt(x).replace('|','/').replace('\n',' ') for x in values)+' |')
    return '\n'.join(lines)


CSS='''body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1500px;margin:35px auto;padding:0 25px;color:#243245;line-height:1.7;background:#fafbfc}h1,h2,h3{color:#143d67}h2{border-bottom:1px solid #ced9e5;padding-bottom:6px;margin-top:35px}table{border-collapse:collapse;width:100%;font-size:13px;background:white}th{background:#244c72;color:white;text-align:left;position:sticky;top:0}th,td{padding:7px 9px;border-bottom:1px solid #e0e6ed;vertical-align:top}tbody tr:nth-child(even){background:#f3f6fa}td.num{white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}figure{margin:25px 0}img{max-width:100%;background:white}details{margin:18px 0;border:1px solid #d4deea;border-radius:4px;background:white;padding:10px}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto;max-height:650px}.filter{width:100%;padding:10px;box-sizing:border-box;margin:10px 0;border:1px solid #afc0d1}.note{background:#edf3f8;border-left:4px solid #4476a6;padding:12px 18px}code{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}a{color:#1c66a0}.meta{color:#596e82;font-size:13px}pre{white-space:pre-wrap;background:#eef3f8;padding:15px}'''
FILTER='''<script>document.querySelectorAll('.filter').forEach(e=>e.addEventListener('input',()=>{let s=e.value.toLowerCase();e.parentElement.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(s));}));</script>'''


def html_table(frame,title=None):
    out=['<section>']
    if title:out.append('<h3>'+html.escape(title)+'</h3>')
    out.append('<input class="filter" placeholder="筛选：输入配置、字段名、类别或关键词">')
    out.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in frame.columns)+'</tr></thead><tbody>')
    for row in frame.itertuples(index=False,name=None):
        out.append('<tr>'+''.join('<td'+(' class="num"' if isinstance(v,(int,float,np.number)) else '')+'>'+html.escape(fmt(v))+'</td>' for v in row)+'</tr>')
    out.append('</tbody></table></div></section>')
    return ''.join(out)


def document(title,body):
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>'+CSS+'</style></head><body>'+body+FILTER+'</body></html>'


def image_block(name,title):
    encoded=base64.b64encode((OUT/'figures'/(name+'.png')).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(title)}</figcaption></figure>'


def graph_metric_definitions():
    d={
        'MEX':('修复边容量上限，本轮 48。','边/节点'),
        'node_count':('全图节点数，本轮 2,505,000。','节点'),
        'nodes_with_repair':('最终修复出度>0 的节点数。','节点'),
        'nodes_without_repair':('最终修复出度=0 的节点数。','节点'),
        'nodes_below_MEX':('最终修复出度<MEX 的节点数，含零度。','节点'),
        'nodes_at_MEX':('最终修复出度=MEX 的节点数。','节点'),
        'nodes_above_MEX':('最终修复出度>MEX 的节点数，检验配额异常。','节点'),
        'ratio_nodes_at_MEX':('nodes_at_MEX/node_count。','比例'),
        'ratio_nodes_with_repair_at_MEX':('nodes_at_MEX/nodes_with_repair，零分母为 0。','比例'),
        'total_repair_edges':('全图有效 NGFix+RFix 修复边条目总数。','有向边条目'),
        'avg_repair_degree_all_nodes':('total_repair_edges/node_count。','边/节点'),
        'max_repair_degree':('全图最大修复出度。','边/节点'),
        'duplicate_repair_entries':('同一源节点修复邻接表中的重复目标条目数；不是全图重复 vector 数。','重复条目'),
        'duplicate_repair_entry_ratio':('duplicate_repair_entries/total_repair_edges。','比例'),
    }
    for pct in [50,90,95,99]:d[f'p{pct}_repair_degree']=(f'修复度数直方图累计达到 ceil({pct}/100×node_count) 的最小整数度数；最近秩分位，不是报告连续变量的线性插值。','边/节点')
    frame=read('field_dictionary')
    frame=frame[frame.scope!='trace_repair_degree_summary.csv metric/value'].copy()
    extra=[dict(scope='trace_repair_degree_summary.csv metric/value',field=k,category='全图修复度数',definition=v[0],unit=v[1],code=location(GRAPH,'writeRepairDegreeSummary'),availability='最终图静态审计') for k,v in d.items()]
    for field in ['ndc_ratio','ndc_nn_share','nearest_useful_probe_rate','nearest_useful_nn_cost_share','ancestor_only_nn_cost_share','shared_cur_recovery_share']:
        if not ((frame.scope=='report_derived')&(frame.field==field)).any():
            extra.append(dict(scope='report_derived',field=field,**definition(field)))
    frame=pd.concat([frame,pd.DataFrame(extra)],ignore_index=True)
    save_table('field_dictionary',frame)
    return frame


def build():
    p=read('performance_all_points');m=read('mechanism_cases').set_index('case').loc[ORDER].reset_index()
    outcomes=read('probe_outcomes');ancestry=read('ancestry_cost_groups');paths=read('baseline_path_groups')
    stats=read('all_field_distributions');class_stats=read('all_class_field_distributions');cats=read('all_categorical_distributions')
    trends=read('feature_trends');dictionary=graph_metric_definitions();checks=read('validation_checks');inv=read('source_inventory')
    for field in ['probes','nn_ndc']:
        outcomes[field+'_share']=outcomes[field]/outcomes.groupby('case')[field].transform('sum')
    outcomes['useful_rate']=outcomes['group'].eq('SUCCESS_RECOVERED_GT').astype(int)
    ancestry['nn_ndc_share']=ancestry.nn_ndc/ancestry.groupby('case').nn_ndc.transform('sum')
    paths['recovered_share']=paths.recovered_gt/paths.groupby('case').recovered_gt.transform('sum')
    save_table('probe_outcomes_with_shares',outcomes);save_table('ancestry_cost_groups_with_shares',ancestry);save_table('baseline_path_groups_with_shares',paths)
    useful=outcomes[outcomes['group']=='SUCCESS_RECOVERED_GT'].set_index('case')
    m['nearest_useful_probe_rate']=[useful.loc[c].probes_share for c in m['case']]
    m['nearest_useful_nn_cost_share']=[useful.loc[c].nn_ndc_share for c in m['case']]
    anc=ancestry[ancestry['group']=='ANCESTOR_ONLY_USEFUL'].set_index('case')
    m['ancestor_only_nn_cost_share']=[anc.loc[c].nn_ndc_share for c in m['case']]
    m['shared_cur_recovery_share']=[paths[(paths['case']==c)&(paths['group']=='SHARED_CUR_MID_NOT_EXPANDED')].recovered_share.iloc[0] for c in m['case']]
    save_table('mechanism_cases_enriched',m)
    cov=dict(source_files=len(inv),performance_files=int((inv.suite=='performance').sum()),mechanism_files=int((inv.suite=='mechanism').sum()),
        performance_points=len(p),cases=len(m),probe_records=int(outcomes.probes.sum()),jump_records=int(sum(json.loads(path.read_text(encoding='utf-8'))['jumps'] for path in MECH.glob('*_k*/ef*/features/feature_manifest.json'))),
        source_column_names=int(inv.header.dropna().str.split(' | ',regex=False).explode().nunique()),dictionary_entries=len(dictionary),
        numeric_distributions=len(stats),class_distributions=len(class_stats),categorical_frequencies=len(cats),checks=len(checks),checks_passed=int(checks.passed.sum()),
        exact_quantiles=True,mechanism_gib=float(inv[inv.suite=='mechanism'].bytes.sum()/1024**3))
    save_json(OUT/'coverage_manifest.json',cov)
    sections=[]
    def section(title,text,frame=None,figure=None):
        sections.append(dict(title=title,text=text,frame=frame,figure=figure))
    section('1. 数据范围、文件完整性与结果目录',
        f'本轮必需输入只有性能目录 graph_k_original_ngfixgrid_1k_v1 和机制目录 bound_features_selected_1k_v1，以及当前 ngfixlib/test/analysis 源码用于定义核对。其他旧实验文件夹不混入。'
        f'性能 {cov["performance_files"]} 文件、机制 {cov["mechanism_files"]} 文件，总计 {cov["source_files"]} 个；机制明细 {cov["mechanism_gib"]:.3f} GiB。'
        f'144 个性能点和 8 组机制诊断都已完成。实际独立查询只有相同的前 1000 条，并非各配置贡献新的查询。'
        f'全量读取 {cov["probe_records"]:,} 条 probe 记录和 {cov["jump_records"]:,} 条已接纳 jump；另有等量 probe 状态行。'
        f'完成 {cov["checks"]} 项核对，{cov["checks_passed"]} 项通过：性能逐查询时间/Recall/NDC，诊断与性能逐查询结果，NDC 与 probe 分解，事件计数，GT 唯一来源归因，probe/state 全行键对齐。'
        '每一个源文件都列在 tables/source_inventory.csv，全部原有小统计表列在《原始统计表汇编.html》。大明细不复制一份，但所有行用于精确统计。',
        inv.groupby('suite').agg(files=('path','size'),bytes=('bytes','sum')).reset_index())
    manifests=json.loads((OUT/'analysis_context.json').read_text(encoding='utf-8'))['manifests']
    base=manifests['hnsw_k10']['inputs']['index']['bytes'];repaired=manifests['ngfix_k10']['inputs']['index']['bytes']
    verify=json.loads((MECH/'index_pair_verification.json').read_text(encoding='utf-8'))
    node_count=verify['base_header']['n'];base_edges=verify['base_stats']['base_edges'];repair_edges=verify['repaired_stats']['repair_edges']
    section('2. 图、距离、查询和计时协议',
        '两图均为 2,505,000 节点、512 维 float 向量，M=16、M0=32、MEX=48，共用入口 1103413。Bottom 为独立未修复底层图；NGFix 保留完全相同的原始邻接顺序，并增加 NGFix/RFix 修复边。查询时不修图。'
        'IP 距离为 1−dot；范数审计：基础图抽样 10k，query 全文件 10k，均在 0.001 的单位范数容差内，没有非有限样本。不能据此证明全部 2.505M 基础向量严格单位范数，也不能套欧式三角不等式。'
        'CPU=1 是逻辑 CPU 编号，实际查询串行单线程；原始硬件型号未记录。timing_protocol=original，gettimeofday 包围每次搜索，计时内 NDC，计时后 GT/距离比；QPS=N×1e6/Σlatency_us。'
        '没有额外显式预热，但计时前做正确性核对，不能称为复现论文冷缓存状态。只有一轮，无法给出轮次间置信区间。诊断带来源和状态采集，不能用其 qps 替代性能 qps。'
        'rderr 列沿用源码的平均距离比，未减 1，精确结果应接近 1；不要把它当作论文公式接近 0 的误差。'
        '本轮只对照相同 M16 底层快照；论文的系统 HNSW 基线 M32/层次结构、构建与修复耗时、负载漂移成本均未重测。',
        pd.DataFrame([dict(graph='HNSW Bottom',nodes=node_count,base_edges=base_edges,repair_edges=0,average_degree=base_edges/node_count,index_bytes=base),
                      dict(graph='NGFix',nodes=node_count,base_edges=base_edges,repair_edges=repair_edges,average_degree=(base_edges+repair_edges)/node_count,index_bytes=repaired)]))
    section('3. 性能总体判断与公平比较',
        '在能找到同图、同 K、recall 不低的 beam 实测点的 54 个二跳配置上，全部 beam 更快。其余 18 个二跳点超出该图 beam 已测 recall 范围，不能外推裁定。'
        '这说明当前不限制成本的二跳尚未显示净查询收益；同 ef recall 增加只能证明额外探索能补回近邻，不能证明效率。'
        '36 个原图二跳点全部存在一个 recall 不低且更快的 NGFix beam 点；但未测构建/维护交换成本，不能只据查询性能裁定完整系统成本。'
        '匹配表保留两边实际 recall 和 ef，不做未测点插值，也不将差距较大的 target-recall 比较冒充严格同 recall。',
        read('matched_recall_points').groupby(['graph','k']).agg(comparable=('beam_reached','sum'),total_bound=('beam_reached','size'),
            min_beam_speedup=('beam_speedup','min'),max_beam_speedup=('beam_speedup','max')).reset_index())
    for k in [10,100]:
        section(f'4.{1 if k==10 else 2} Recall@{k}、QPS、NDC 和尾部延迟',
            '图中的 P95 来自本轮原始逐查询搜索延迟。NDC 的 P95/P99 是距离计算分布，不能当作延迟分位。所有点的完整表见性能附件；性能结果不使用诊断计时。',
            p[p.k==k][['graph','method','ef','recall','qps_median','mean_ndc','ndc_p95','latency_p95_us','latency_p99_us']],
            f'performance_k{k}')
    section('5. 相同 ef 的机制收益、损失与 NDC',
        '每组在相同 q、同图、同 K、同 ef 上配对。设 B=beam 最终结果、T=二跳最终结果、GT=精确 top-K：'
        'recovered=|(T∩GT)−B|，lost=|(B∩GT)−T|，Δrecall=(recovered−lost)/K。'
        '直接 recovered 优先归为 jump root 本身；其余为最近 root 后继。非 GT 中转是路由的子集，不能再相加。K=10 与 K=100 的 GT 标签不同。',
        m[['case','paired_baseline_recall_mean','recall_mean','delta_recall','ndc_mean','paired_baseline_ndc_mean','ndc_ratio',
           'recovered_gt_total_mean','lost_gt_total_mean','improved_fraction','worsened_fraction','direct_recovered_share','routing_recovered_share','non_gt_router_recovered_share']],
        'gt_recovery_and_loss')
    section('6. 成本究竟花在哪里',
        'ndc=1+ndc_mid+ndc_nn；ndc_nn=ndc_nn_jump+ndc_nn_nojump。成功 probe 的全部 fresh NN 成本归入 nn_jump，不只是入队 NN。'
        '没有接纳任何候选的 probe 占二跳 NDC 的 76.43%–90.82%；NN 本身又占总 NDC 的大部分。visited/重复扫描比例为 14.18%–34.74%，但绝大部分失败不是“没有 fresh NN”，而是评分后没有一个通过 beam。'
        '最近来源定义的 useful probe 比例仅 0.0144%–1.556%；不能把成功接纳率 7%–20% 当作真正有效率。'
        '“SUCCESS_NO_RECOVERED_GT”只表示没有最近归因新增，不是反事实可删除标签。祖先-only probe 占一些恢复路径的上游依赖，已另外统计。即便没有观测 recovered 祖先，也不保证删除后轨迹不变。',
        m[['case','ndc_nn_share','successful_probe_rate','no_admission_nn_cost_share','duplicate_or_visited_scan_fraction',
           'nearest_useful_probe_rate','nearest_useful_nn_cost_share','ancestor_only_nn_cost_share','net_gt_per_1k_nn_ndc']],
        'ndc_cost_decomposition')
    section('5.1 有丢失不等于净 recall 下降',
        'HNSW K100 ef100 有 56% 的 query 至少丢失一个 baseline GT，但只有 0.2% query 的净 recall 下降；'
        '这些 query 通常同时补回更多 GT。不能用 worsened_fraction 代替 lost_any，也不能只报 recovered 忽略 lost。',
        read('query_recovery_loss_overlap'))
    section('6.1 所有 probe 的四类分布与成本',
        'FAILED_NO_FRESH 的 NDC 为零但仍有邻接扫描、visited、锁/容器等成本。FAILED_WITH_FRESH 是主导浪费。'
        'SUCCESS_RECOVERED_GT 和 SUCCESS_NO_RECOVERED_GT 使用最近 root 归因，不是因果 oracle。完整祖先视图见下一表。',
        outcomes[['case','group','probes','probes_share','nn_ndc','nn_ndc_share','admitted','recovered_gt']],
        'probe_outcomes')
    section('6.2 祖先-only 贡献不能遗漏',
        'NEAREST_USEFUL 有直接或最近后继新增 GT；ANCESTOR_ONLY_USEFUL 没有最近归因但其 root 是后续恢复链的祖先；'
        'NO_RECOVERED_ANCESTRY 没有记录到恢复祖先贡献。后两者不是同一类。ancestor 数不能相加作唯一 GT；这里只对“是否有贡献”做布尔分组。',
        ancestry[['case','group','probes','nn_ndc','nn_ndc_share','recovered_gt']])
    section('7. 已有短路径被屏蔽，还是发现了新上游区域',
        'SHARED_CUR_MID_NOT_EXPANDED 表示 baseline 扩展过 cur、评分过 mid、未扩展 mid；这部分 uniquely attributed recovered 占 32.36%–84.63%。'
        'NEW_UPSTREAM_CUR 表示 baseline 没扩展 cur，说明连续二跳/顺序变化带来了新上游，而非一次共同前沿的简单救援。'
        'BASELINE_MID_EXPANDED 的最近归因 recovered 在本轮均为 0。'
        '这支持“原图已有短路径未被普通搜索充分利用”是部分收益机制，但不能将全部收益解释为同一状态下 baseline 明确拒绝同一个 mid；未扩展也可能受停止/排序影响。',
        paths[['case','group','probes','nn_ndc','recovered_gt','recovered_share']],
        'baseline_path_recovery')
    section('8. 边类型、局部图位置与邻居几何',
        'H=原始边、N=NGFix、R=RFix。分别统计 examined 邻接项、fresh-scored、admitted、unique recovered；分母不能互换。'
        '原图全部 HH；修复图 NN 占 fresh NN 成本 43.62%–49.15%，但占 unique recovered 51.29%–63.17%；不是只靠原图 HH 路径。'
        'RR 等罕见组合有时 ROI 数值很高，但绝对恢复只有 1–2 个，不能据此制定全局规则。'
        'node ID 不是空间位置；发现路径深度不是最短路；BASE_ONLY/MIXED/REPAIR_ONLY 是出边组成，不是社区检测。'
        'reverse/direct-edge 类型查询返回首个匹配边，重复/平行边不展开成所有类型集合；实际两条遍历边用记录的 first/second 类型。',
        read('edge_pair_recovery'),
        'edge_pairs_cost_and_recovery')
    section('8.1 Jaccard 原有分桶的退化与修正解释',
        'HNSW 的 cur-mid Jaccard 中位数和 P75 为 0，唯一值仍有 166–212 个；原 qcut 四分桶因边界重复而退化成一桶。这不是恒定特征。'
        '零/非零对照显示，8 组中存在共同出邻居的 probe 均有更高最近归因有效率和每千 NN 距离恢复产出。'
        '它是局部结构关联，既不能推断社区边界，也不说明越高 Jaccard 在所有范围都更好。',
        read('sparse_jaccard_controls'),
        'jaccard_zero_nonzero')
    section('9. 搜索状态特征的规律',
        'DARTH 风格快照在 mid 评分/拒绝后、任何第二跳评分前获取。保留当前步数、NDC、ef/top-K 更新、距上次 top-K 更新步数、发现深度、队列大小以及两组距离分布。'
        '8/8 配置低 mid_gap_normalized 桶更有效，高 cur_slack_normalized 桶更有效，较短 top-K 停滞更有效；低 gap 桶对比高 gap 的有效率提高约 12–29 倍。'
        '因此本轮不支持先预设“越停滞越该探测”；持续停滞的 probe 通常更差。'
        '7/8 配置较早 step/NDC 更有收益，HNSW K10 ef20 例外；发现深度高值更有效不等于搜索越晚越好，深度和全局展开步不是同一变量。'
        'cur 低度桶和 mid 高度桶有效率较高，但 mid 高度常有更差成本 ROI。frontier_size、更新次数方向不统一；距离均值/方差/分位数互相相关，不能把所有关联当独立信号。'
        '所有变量逐配置的低/高桶、有效率、查询平衡率、ROI 和完整分布都保留。下面是首尾桶描述，不是显著性检验或已验证的组合触发策略。',
        trends,
        'feature_associations')
    section('9.1 搜索阶段',
        '20 个离线阶段以最终总展开次数归一化；不是在线可用时钟。阶段表中的 recovered 只按最近来源计，不扣 lost；'
        '必须同时看 probe 数和 fresh NN 成本，不能只看成功 jump 出现的阶段。阶段曲线整体显示后段收益衰减，但不同图、K、ef 仍有差异。',
        read('search_phase_bins'),
        'search_stage_yield')
    section('10. K、冗余特征、固定困难组与节点复用',
        '同图 ef100 的 K10/K100 probe 路径键哈希完全一致，NDC、probe、jump 数也相同；只是返回 K、GT 标签、top-K 状态和后继标签不同。不能把两个 K 当两套独立搜索轨迹。'
        'beam_size=ef、topk_size=K 在本轮 probe 时恒定；同一查询 entry_dist 恒定；topk_min=beam_min；K=ef 时 top-K/beam 距离分布重复。'
        '固定困难组以 HNSW beam ef200 的最低 recall 20% 选出（同分按 NDC 高、query_id 小排序），再跨图/方法/ef固定 IDs。其子组 QPS 由原逐查询时间重建，不是另一次独立测速。'
        'node reuse 分别按 cur/mid/nn 和 jump_class 计算 top1% 集中度；热门节点、共享 query 和 GT 可能导致集中，不等同于“存在可识别跳跃社区”。',
        read('same_ef_k_trace_checks'))
    group=read('fixed_query_group_performance')
    section('10.1 同一批困难查询的表现',
        '以下是各 K 自己固定的困难 20%（200 query IDs），ef100 的四方法对照。'
        'K100 原图 recall 0.38515→0.64580，NGFix 0.81040→0.95690；二跳确实帮助难查询，但 NDc 与时间成本也显著增加。'
        '该组 QPS 由原始逐查询计时重建，不能作为新轮次或独立负载测量；完整 ef/全部查询组表保留在 CSV 与 Excel。',
        group[(group.ef==100)&(group['group']=='fixed_low_recall_20pct')][['graph','k','method','queries','recall','mean_ndc','latency_mean_us','reconstructed_qps']])
    section('11. 全图审计、范数、EH 与常量字段',
        f'NGFix 新增 {repair_edges:,} 有向修复边，比原始有向边多 {100*repair_edges/base_edges:.2f}%；索引文件仅增加 {100*(repaired/base-1):.2f}%（{(repaired-base)/1024**2:.2f} MiB），因为共用的大向量区占主体。'
        '修复节点 2,212,122（88.31%），修复度数中位数10、P90/P95/P99=48；全体节点满 MEX 比例10.87%，重复修复条目202（约5.19 ppm）。'
        '图 EH 排除 RFix，测试时只读取历史修复属性；EH_INF 的65535、RFix的201不是连续当前查询难度。'
        'degree_lt/eq_MEX 是最终图饱和代理，不重建插边当时是否已满。'
        'dabs_gamma=NaN 是本轮不适用。关闭 gate/cheap/full 选择相关字段为0不说明搜索没有进展；真实进展看 state_*。'
        '无 best-scored NN 的占位结构列在报告分布中置为不适用，未把原 CSV 的占位0当作真实零度节点。',
        read('graph_metrics'))
    section('12. 每个字段、全部数值分布和原始统计表在哪里',
        f'《字段与数值分布.html》包含 {len(dictionary)} 条按表语境定义的字段/图指标，覆盖 {cov["source_column_names"]} 个源 CSV 列名及额外派生特征。'
        f'包括 {len(stats)} 组全量数值分布、{len(class_stats)} 组分类数值分布、{len(cats)} 条类别/小基数频数。'
        '所有分位数精确计算，使用线性插值；修复度数审计沿用离散最近秩。均值、标准差、最小、P1/P5/P25/P50/P75/P90/P95/P99、最大、缺失、零值、唯一值数均记录。'
        '事件/probe 分布按记录加权，困难查询贡献更多记录；另保留按 query 平衡的 useful rate，但各特征桶出现 query 集合不同。'
        '《原始统计表汇编.html》完整收录原小统计表的所有行和列，重复图审计按各源路径保留；source_inventory 明确每个 CSV/JSON/log 是否读取。'
        '原 8.86 GiB 大明细保留在输入目录，报告不重复复制每个事件。Performance/query/latency 原始行仍在原目录，所有行已用于本报告的分布和配对核对。')
    section('13. 目前能下的结论与下一轮验证',
        '成立：当前无限二跳在已测可比较区间没有净 QPS 收益；距离成本主要集中于不接纳 probe；补回来源兼有共同上游未展开 mid 的路径和新上游探索；近边界 mid、较大 cur slack、较短停滞及邻居重合有跨配置的观察关联。'
        '尚未成立：只保留 useful 或低 gap 的 probe 就能保持 recall；这些标签使用最终 GT 和祖先关系，筛掉某次探测会改变后续 visited/beam/路径。'
        '建议冻结低成本规则后，用独立 query IDs 与同预算随机/邻接序对照；评价同 recall QPS、全部 NDC、计算/缓存特征成本和损失。按 query 划分训练/验证，不能随机拆同一 query 的 probe。'
        '限制：第一批1k查询不是独立验证集；没有多轮 QPS、不记录CPU温度频率；诊断缺独立 binary SHA/source commit，当前按代码与输出核对；未测维护/修复/漂移成本；图位置是局部描述而非社区；所有 source/后继统计是关联与来源，不是严格反事实因果。')
    md=['# HNSW / NGFix 搜索实验完整报告','', '报告日期：2026-10-13。全部数值来自本地原 CSV，未使用控制台取整值替代。分位数是对保存值的全量精确计算；事件/结构原 CSV 本身的打印精度不能被恢复成内部浮点精度。','']
    body=['<h1>HNSW / NGFix 搜索实验完整报告</h1><p class="meta">2026-10-13 · 原始 CSV 全量分析 · 1k 查询 · original 计时</p>',
          '<p><a href="特征逐项解读.html">36 个特征逐项解读</a>　<a href="字段与数值分布.html">字段与全部分布</a>　<a href="原始统计表汇编.html">全部原始小统计表</a>　<a href="实验数据汇总.xlsx">Excel 汇总</a></p>']
    def chapter_order(s):
        m=re.match(r'(\d+)(?:\.(\d+))?',s['title'])
        return (int(m.group(1)),int(m.group(2) or 0)) if m else (999,0)
    for s in sorted(sections,key=chapter_order):
        md.extend(['## '+s['title'],'',s['text'],''])
        body.extend(['<h2>'+html.escape(s['title'])+'</h2><p>'+html.escape(s['text'])+'</p>'])
        if s['figure']:
            md.extend([f"![{s['title']}](figures/{s['figure']}.png)",'']);body.append(image_block(s['figure'],s['title']))
        if s['frame'] is not None:
            md.extend([md_table(s['frame']),'']);body.append(html_table(s['frame']))
    (OUT/'完整报告.md').write_text('\n'.join(md),encoding='utf-8')
    (OUT/'完整报告.html').write_text(document('完整实验报告',''.join(body)),encoding='utf-8')
    # Exhaustive appendices are separate so the narrative is still navigable.
    appendix=['<h1>字段定义与全部数值分布</h1><p><a href="完整报告.html">返回主报告</a></p>',
        '<p>每个字段有定义、单位、可用时点和代码位置。分布按各配置/表语境计算；标识列数值分位不作为几何特征。源码占位值与不适用已区分。记录级分布不等于查询等权分布。</p>']
    for title,frame in [('字段手册',dictionary),('全部数值分布',stats),('按成功/失败/jump 类别的全部数值分布',class_stats),('类别及小基数频数',cats),('特征分桶和 ROI',read('probe_feature_bins')),('零/非零 Jaccard 对照',read('sparse_jaccard_controls')),('356 项一致性核对',checks),('全部源文件清单',inv.drop(columns=['header'],errors='ignore'))]:
        appendix.append('<details><summary>'+html.escape(title)+f'（{len(frame):,} 行）</summary>'+html_table(frame)+'</details>')
    (OUT/'字段与数值分布.html').write_text(document('字段与数值分布',''.join(appendix)),encoding='utf-8')
    originals=json.loads((OUT/'existing_tables_all.json').read_text(encoding='utf-8'))
    rows=['<h1>全部原始小统计表汇编</h1><p><a href="完整报告.html">返回主报告</a></p><p>按原配置与源表保留全部列，不把不同单位的 value 列混合平均。大事件表的全部字段与分布在另一附录；原文件路径见清单。</p>']
    grouped={}
    for r in originals:grouped.setdefault((r['case'],r['table']),[]).append(r['values'])
    for (case,table),values in grouped.items():
        frame=pd.DataFrame(values);rows.append('<details><summary>'+html.escape(case+' / '+table)+f'（{len(frame):,} 行）</summary>'+html_table(frame)+'</details>')
    # Performance-derived input summaries are likewise preserved.
    for path in sorted((PERF/'comparison').glob('*.csv')):
        frame=pd.read_csv(path);rows.append('<details><summary>'+html.escape('performance comparison / '+path.name)+'</summary>'+html_table(frame)+'</details>')
    for path in sorted(PERF.glob('*_k*/clean/analysis/*.csv')):
        frame=pd.read_csv(path);rows.append('<details><summary>'+html.escape(str(path.relative_to(PERF)))+'</summary>'+html_table(frame)+'</details>')
    for suite,root in [('performance',PERF),('mechanism',MECH)]:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ['.json','.log','.txt']:
                text=path.read_text(encoding='utf-8',errors='replace')
                rows.append('<details><summary>'+html.escape(suite+' / '+str(path.relative_to(root)))+'</summary><pre>'+html.escape(text)+'</pre></details>')
    (OUT/'原始统计表汇编.html').write_text(document('原始统计表汇编',''.join(rows)),encoding='utf-8')
    # Workbook data: distinct analytical views, not a copy of >1m-row raw traces.
    workbook=[]
    specs=[('Performance','完整性能曲线','performance_all_points'),('RecallMatches','不低 recall 的实测 beam 对照','matched_recall_points'),('RecallTargets','目标 recall 的实测匹配','target_recall'),('MechanismCases','同图同 ef 配对机制','mechanism_cases_enriched'),
        ('ProbeOutcomes','全部 probe 的接纳与最近归因成本','probe_outcomes_with_shares'),('AncestryCosts','最近与祖先贡献成本','ancestry_cost_groups_with_shares'),
        ('BaselinePaths','共同前沿与新上游的恢复来源','baseline_path_groups_with_shares'),('EdgePairs','九种边对的机会、成本与归因恢复','edge_pair_recovery'),
        ('FeatureTrends','每个特征逐配置的首尾桶观察关联','feature_trends'),('FeatureBins','全部特征桶的有效率和 ROI','probe_feature_bins'),
        ('Distributions','所有原始数值字段的精确分布','all_field_distributions'),('ClassDistributions','成功失败与 jump 类别的精确分布','all_class_field_distributions'),
        ('Categories','类别与小基数字段的完整频数','all_categorical_distributions'),('Definitions','所有字段与图指标的代码定义','field_dictionary'),
        ('SearchStages','20 个离线阶段的机会和收益','search_phase_bins'),('Visibility','jump 在 paired beam 中的可见性','jump_visibility'),
        ('SparseJaccard','邻居重合零值与非零值对照','sparse_jaccard_controls'),('FixedQueryGroups','固定困难查询组对照','fixed_query_group_performance'),
        ('GraphMetrics','最终全图修复度数审计','graph_metrics'),('Validation','独立一致性核对','validation_checks'),('Inventory','全部源文件目录','source_inventory')]
    for name,title,table in specs:
        frame=read(table)
        if name=='Inventory':
            frame=frame.drop(columns=['header'],errors='ignore')
            # Keep nanosecond metadata exact across JSON/JavaScript/Excel.
            frame['mtime_ns']=frame.mtime_ns.map(lambda v:str(int(v)))
        workbook.append(dict(name=name,title=title,columns=frame.columns.tolist(),rows=frame.to_numpy(dtype=object).tolist()))
    save_json(OUT/'workbook_data.json',dict(sheets=workbook,coverage=cov,summary_cases=ORDER,
        notes=['1000 条相同查询、单轮 original 计时。性能与带统计诊断时间分开。',
               '字段和精确分布完整；原始大事件表保留在输入目录，不复制到 Excel。',
               '统计关联不是因果删除规则。NaN/无候选不当真实零。',
               '源文件与代码位置见 Inventory/Definitions；全部原始小统计表见 HTML 附录。']))
    print('Narrative and exhaustive appendices saved; definitions',len(dictionary),'checks',len(checks),flush=True)


if __name__=='__main__':build()
