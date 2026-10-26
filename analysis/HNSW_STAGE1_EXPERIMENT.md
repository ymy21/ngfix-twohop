# 第一阶段：未修复 HNSW 底层图上的净收益与机制

## 本轮问题与固定条件

先回答：原图上的不受成本限制的 bound-twohop，在达到同一 Recall@100
目标时，是否比普通 beam search 增大 ef 更快？更高 recall、相同 ef 或
更少第一跳 NDC，都不能单独证明净收益。

固定已验证的 Bottom 文件、节点 ID、向量、入口 1103413、IP 距离、
同一份 query.10k.fbin/gt.10k.ibin、K=100、一个搜索工作线程与 CPU 绑定。
算法简称为“底层图 beam search”，当前实验不代表完整多层 HNSW 导航。

方法 A：普通 beam search。方法 B：默认 bound-twohop，即 immediate
调度、拒绝 mid 后立即完整扫描 N(mid)、按精确距离排序二跳候选，使用
实时 beam 条件接纳候选。关闭所有成本上限、gap/cur/历史/tree 筛选；
保留正常 visited 去重、终止条件和 beam 剪枝。不是全图无条件二跳。

已有 NGFix 二跳方法、clean32/NN8 测速入口和全部旧脚本都保留。
新增干净实现只支持上述固定默认语义，避免将默认语义改成 after_onehop。

## A：性能与 NDC 曲线

ef 网格：100,120,150,180,200,250,300,400,500,600,800,1000,1200,1600,2400。
两种方法使用相同初始网格，但最终比较允许不同 ef。

干净计时中只有搜索、结果返回和每查询固定大小的校验和累加；不做
GT 对照、NDC 计数、逐边分类、来源记录、逐查询计时、文件写入。
NDC 计数在模板实例化时关闭。图/查询加载与 GT 整理不在计时中。
每个点预热一遍；5 轮随机打乱方法/ef 点顺序；每个批次至少 1 秒。
QPS=批次查询数/批次耗时，输出中位数、四分位范围、最小值、最大值。
这是单工作线程、预热后的批量吞吐率；不能从其反推 P95/P99 延迟。

Recall 和 NDC 在计时外对全体查询逐条计算；NDC 包括入口完整距离，
不包括离线真值评估。每个点默认前 200 条查询与旧的参考算法对比完整
结果及 NDC；全体查询检查有计数与无计数内核返回结果相同。
参考不一致则中止，不生成已完成的比较。

主要输出：

- clean/analysis/recall_qps_ndc_curve.csv：每方法/ef 的 Recall、干净 QPS、
  NDC 均值及 P50/P95/P99。NDC 分位数是成本分布，不是延迟分位数。
- clean/analysis/at_target_recall.csv：0.90/0.95/0.97/0.98/0.99/0.995
  各目标上，选达到目标的最快实测点，保留实际 recall 与 ef。
  相差超过 0.001 时提示加密 ef；未达到目标时留空，不做外推。
- clean/timings.csv.queries.csv：逐查询 Recall/NDC。
- clean/analysis/paired_query_groups.csv：同 ef 配对变化，全体查询与固定
  困难组。困难组取 baseline ef=200 的最低 recall 20%，平局按 NDC
  从高到低、再按查询 ID 排序；跨方法/ef 不重新分组。
- clean/manifest.json：输入摘要、查询/GT SHA256、二进制 SHA256、参数。
- 安装 matplotlib 时额外生成 recall_qps_ndc.png；CSV 无额外依赖。

QPS 胜负以接近 recall 的实测点和重复轮次为依据。初始网格仅用于定位；
在目标两侧增加 ef 点，再复测。不要将高 recall 但更慢说成净收益，也
不要把 round min/max 当成置信区间。

## B：配对机制统计，完全不使用诊断 QPS

独立运行原 search_hnsw_ngfix 的 boundtwohopcmp：先对每条查询跑
baseline，再跑 bound-twohop。全体 10k 查询在全部 ef 上记录配对
Recall/NDC、recovered/lost GT、距离计算分区与探测收益。
全量诊断逐查询 Recall/NDC 必须与干净测速的计时外记录一致，否则
后处理报错，不能把两套入口的统计混合使用。

机制成本重点：ndc = ndc_entry + ndc_mid + ndc_nn；同时看总 NDC 差，
因为减少 mid NDC 可能只是把距离计算提前到了 nn。

summary/paired_mechanism_summary.csv 输出：

- 原始/二跳 recall 与总 NDC、改善/下降查询比例；
- 新增 GT、丢失 GT、净新增 GT；
- 新增 GT 的直接跳入、后继路由、非 GT 中转根贡献与未归因部分；
- 至少接纳一个新候选的探测比例；
- 访问邻接项中的已访问/重复目标比例；
- 没有接纳任何候选的探测消耗了多少二跳完整距离；
- 非 HH 路径数量：Bottom 运行必须为 0，否则视为实验污染。

非 GT 中转根贡献是路由贡献的子集，不可与直接/路由三者相加。
有接纳候选只代表局部成功，不代表最终 recall 增益。

进一步在 ef=100/200/400/800、前 1000 条固定查询采集事件。
无事件查询仍纳入 paired summary 分母。事件按两条轴分类：

1. DIRECT_GT_NEW / DIRECT_GT_SHARED / ROUTER_TO_GT / ROUTER_NO_GT。
2. baseline 未评分 / 已评分未扩展 / 已扩展该 jump 节点。

将 DIRECT_GT_NEW 中 baseline 未评分的比例，与新增 GT 的直接/路由
贡献结合，判断是否主要在补救原来没有到达的区域。进一步利用 mid
事件中的 baseline_expanded_cur、baseline_scored_mid、baseline_expanded_mid，
识别 baseline 已到达上游节点但没有扩展中间节点的短路径。

注意：bound 轨迹中 mid 被拒绝，并不证明 baseline 在它自己的搜索
状态下也以相同边界拒绝 mid；baseline 已评分未扩展也可能源于停止
条件。因此这些是轨迹证据，不是严格因果结论。路由 ancestry 属于
来源归因，不能把所有 ancestor 后继数相加作为唯一新增 GT。

## C：与之前 NGFix 结果的机制对照

旧 result/webvid/ngfix_boundtwohopcmp_k100.csv 的列兼容；使用
summarize_hnsw_mechanism.py 的 --ngfix-paired 可生成描述性比较，按
相同 ef/query_id 对齐。先核对相同 query/GT、K、metric、查询范围、
入口、默认 immediate/full 无预算语义及 NDC 包含入口的口径。
历史日志没有完整来源证明时，报告标记为 metadata 未核实。

旧 result/clean_qps1k_20261008_v1 测的是有预算的 cheap32/NN8，且只用
1k 查询；不能与这轮 10k/unbounded 曲线直接归因于图差异。
旧普通二跳诊断 CSV 的 QPS 带统计负担，也不能和新干净 QPS直接比。

比较机制时：Bottom 只有 HH；NGFix 还应分 HH/HRepair/RepairH/RepairRepair。
比较按唯一新增 GT 的直接/路由份额与每千次二跳距离的产出，避免只
比较接纳候选数量。旧 CSV 没有 score_pair 时，不用 examined 代替
fresh-scored 作为精确候选收益分母。

如需严格对照速度，下一阶段在 NGFix 图上用同一个新测速二进制，
同样查询范围、ef、机器和重复轮次重跑；run_hnsw_stage1.py
支持 --allow-repaired，但该运行应另建目录，标为 repaired_reference。
本轮不默认启动这一组。

## 可识别失败机制的判断顺序

先确认至少一个接近 recall 的稳定区间存在干净 QPS 净收益。
若无限二跳均更慢，但确有新增 GT，结论为“存在补救机会，当前无限
策略没有净收益”，下一步仅验证选择性策略能否回收该机会。
若两者都没有，暂缓复杂触发策略。

再检查新增 GT 是否集中在 unseen 的直接补回、非 GT 中转引导等一类，
并同时扣除丢失 GT、量化无接纳探测成本。观察低 mid gap、较大 cur
slack、搜索阶段等可在线计算特征是否区分有收益与无收益探测。
不得用“节点集中度高”单独代替失败机制。

特征选择用前 1000 条诊断查询；冻结规则后，用独立的后 9000 条查询
复核。若在全体 10k 上选规则，后续必须另找独立测试集。
归一化 hop_frac 需要最终总展开次数，只适合作为离线描述，不能直接
作为在线触发特征。GT/useful/baseline 可见性标签同样不能在线使用。

最后做必要消融：规则触发 vs 同预算随机触发 vs 无限二跳 vs 增大 ef；
验证效果集中在预测的失败类型，且有实际净收益。当前第一阶段不
提前加入规则，以免改变待验证的算法。

## 服务器运行

同步新增文件以及 hnsw_ngfix.h、test/CMakeLists.txt 后编译：

```bash
ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target bench_hnsw_unbounded_clean test_unbounded_clean_search search_hnsw_ngfix -j 4
```

先做 200 查询/两个 ef 的功能检查（其结果不作为最终性能证据）：

```bash
OUT="$ROOT/result/hnsw_stage1_smoke_v1" QUERY_LIMIT=200 ROUNDS=1 EF_VALUES=100,200 RUN_DIAGNOSTICS=0 \
  bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
```

确认参考一致后，完整运行；也可以先只运行性能轮，减少等待：

```bash
RUN_DIAGNOSTICS=0 bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
```

完整性能与诊断一次运行（新目录）：

```bash
OUT="$ROOT/result/hnsw_stage1_full_v1" bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
```

脚本拒绝覆盖旧 OUT。不可同时运行两个测速作业。无限二跳可能很慢，
先根据 smoke 日志估计正式运行成本，再安排空闲机器。

历史机制对照示例：

```bash
python3 "$ROOT/analysis/summarize_hnsw_mechanism.py" \
  --paired "$ROOT/result/hnsw_stage1_full_v1/diagnostics/base_paired.csv" \
  --ngfix-paired "$ROOT/result/webvid/ngfix_boundtwohopcmp_k100.csv" \
  --out "$ROOT/result/hnsw_stage1_full_v1/historical_mechanism_comparison"
```

本地 Windows 环境没有 Linux/C++ 编译工具链；C++ 编译与等价性测试
需在服务器执行。启动脚本将先执行 test_unbounded_clean_search，再在
实际数据上验证参考算法，验证失败则停止。
