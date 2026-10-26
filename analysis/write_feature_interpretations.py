"""Per-feature code definition, all eight case distributions and observed patterns."""
from pathlib import Path
import html
import pandas as pd
from write_complete_experiment_report import OUT,read,md_table,html_table,document
from report_field_catalog import definition

NOTES={
 'state_step':'7/8 配置较低步号桶更有效，HNSW K10 ef20 例外。步号与累计 NDC、结果距离、查询难度相关，不能仅以“越早越好”作全局规则。',
 'state_ndc':'7/8 配置低累计 NDC 桶更有效。它主要描述搜索进展/难度，不是本次 probe 成本的预测值。',
 'state_ndc_mid':'与 state_step/state_ndc 强相关，7/8 低值桶更有效；不能把相关指标当多个独立证据。',
 'state_ndc_nn':'7/8 低值桶更有效；高值可能意味着此前已经大量探测而边际机会减少。记录不计本次 probe 的未来成本。',
 'state_result_updates':'5 高值有利、3 低值有利，方向不一致；更新次数受 ef、路径和查询难度共同影响。',
 'state_topk_updates':'4 高值有利、4 低值有利，没有统一方向；GT 目标 K 改变会改变该计数，即使物理搜索轨迹相同。',
 'state_steps_since_topk_update':'8/8 低值桶更有效。观测不支持预设“越长期停滞越应该 probe”；它仍是轨迹关联，需独立规则消融。',
 'state_discovery_depth':'8/8 高值桶更有效，但深度是发现路径长而不是扩展时刻或图最短路。可出现较深路径在较早全局步号扩展。',
 'state_frontier_size':'5 高值有利、3 低值有利；candidate_set 可能保留过期候选，长度不是有效机会数。',
 'state_beam_size':'实际 probe 触发要求 beam 已满，本轮每组恒等 ef。不能在组内区分 useful/failed。',
 'state_topk_size':'本轮每组恒等 K，不能在组内区分 probe。',
 'state_entry_dist':'6 低值有利、2 高值有利，方向不稳定。同一 query 恒定，按 probe 的分布会给困难 query 更多权重。',
 'cur_total_degree':'8/8 低度桶有效率更高，不能据此推断减少图度数会改善算法；当前是固定图内的选择关联。',
 'mid_total_degree':'8/8 高度桶有效率更高，但 NN 扫描/评分更多。多数配置的高值桶成本 ROI 反而更低，需把机会与成本分开。',
 'cur_repair_ratio':'HNSW 的 4 组恒为 0；NGFix 的 4 组高值桶更有效，但与 degree/历史热点混杂。',
 'mid_repair_ratio':'HNSW 恒为 0；NGFix 高值桶更有效，但更大的修复邻接区也增加二跳成本。',
 'cur_mid_data_dist':'6 高值有利、2 低值有利，方向不统一；它是 1−dot，不应套用欧氏三角不等式。',
 'cur_mid_common_neighbors':'原 HNSW 的大量零值让 qcut 退化；零/非零补充对照表明 8 组中有共同出邻居的 probe 更有效。交集按去重邻居集合算，度数却按边条目算。',
 'cur_mid_jaccard':'HNSW 中位数/P75=0 但并不恒定。零/正值对照均显示正值组较好；不证明越高越好或属于同一真实社区。',
 'mid_gap_normalized':'8/8 低值桶更有效，低桶对高桶约 12–29 倍。它是最稳定的单变量观察信号之一，但改变探测后路径/边界也会变，不能把离线收益当上线回收率。',
 'cur_slack_normalized':'8/8 高值桶更有效，与 mid_gap 互补：cur 更在 beam 内部、mid 更接近边界，形成较有希望的二跳机会。联合规则尚未检验。',
 'topk_spread':'6 低值有利、2 高值有利；与 topk_variance 和距离分位数相关，方向并非完全一致。',
}


def run():
    trend=read('feature_trends');stats=read('all_field_distributions');cs=read('all_class_field_distributions')
    features=sorted(trend.feature.unique())
    md=['# 每个探测前特征的逐项解读','', '本附录逐一覆盖全部 36 个 probe 特征。精确值见完整表；首尾桶关联不代表因果、训练结果或反事实性能。','']
    body=['<h1>每个探测前特征的逐项解读</h1><p><a href="完整报告.html">返回主报告</a>　<a href="字段与数值分布.html">全部字段与分布</a></p>',
          '<p>36 个 probe 特征逐一列出。每项包括代码定义、八组精确分布/分类分布和分桶结果。nn 距离/候选排名等评分后特征另在完整字段手册与 jump 分布中，不能当作探测前输入。</p>']
    for field in features:
        d=definition(field)
        note=NOTES.get(field)
        if not note:
            if field.startswith(('beam_','topk_')):
                note='距离水平（min/max/mean/median/q25/q75）通常 7/8 高值桶有更多恢复机会，可能反映较难 query 或尚未收敛；variance 则 6/8 低值更有效。相互相关，topk_min=beam_min；K=ef 时两套分布重复。不能据单变量方向推断因果。'
            else:note='方向与取值、上下文有关；以各组数据为准。'
        text=d['definition']+' 单位：'+d['unit']+'。代码：'+d['code']+'。 '+note
        md.extend(['## '+field,'',text,'']);body.extend(['<h2>'+html.escape(field)+'</h2><p>'+html.escape(text)+'</p>'])
        raw=stats[stats.field==field]
        if len(raw):
            raw=raw[['case','file','valid','missing','unique','mean','std','min','p25','median','p75','p95','p99','max']]
            md.extend([md_table(raw),'']);body.append(html_table(raw,'全量数值分布'))
        else:
            md.append('此字段由原始状态/距离派生，其精确桶边界与原分类分位见下表及原统计表汇编；未声称恢复未保存的更多浮点精度。\n')
        select=trend[trend.feature==field][['case','bins','low_bin','high_bin','low_useful','high_useful','low_query_balanced','high_query_balanced','low_roi','high_roi','pattern']]
        md.extend([md_table(select),'']);body.append(html_table(select,'完整低/高桶与成本产出'))
        classes=cs[cs.field==field]
        if len(classes):body.append('<details><summary>按成功/失败/jump 类别的完整分布</summary>'+html_table(classes)+'</details>')
    (OUT/'特征逐项解读.md').write_text('\n'.join(md),encoding='utf-8')
    (OUT/'特征逐项解读.html').write_text(document('逐项特征解读',''.join(body)),encoding='utf-8')
    print('Feature interpretations:',len(features))


if __name__=='__main__':run()
