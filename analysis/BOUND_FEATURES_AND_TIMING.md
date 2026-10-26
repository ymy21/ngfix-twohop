# 计时协议与 jump 特征扩展

## 两种明确区分的计时协议

默认仍是 TIMING_PROTOCOL=batch：steady_clock 包围整批搜索；NDC 在
计时外计算，计时内关闭 NDC、机制记录、GT 评估和输出；每轮预热，
随机打乱方法/ef 测试点，批次至少 1 秒。

新增 TIMING_PROTOCOL=original：gettimeofday 包围每次搜索调用；
beam 直接调用未改的 searchKnn()，二跳调用结果和 NDC 等价的最小
计数内核 searchKnnBoundUnboundedClean<true>()。两者在计时内计数
NDC，不收集机制来源。GT 评估和距离比计算在计时后、下一查询之前
进行；延迟 CSV 在一个测试点的全部查询结束后写出。
QPS = N*1e6/sum(latency_us)，不是 mean(1e6/latency_us)。
original 使用顺序 ef 点和顺序查询，无额外显式预热；rounds 仍可重复。
min-batch-ms 不适用，分析器按协议区分验证。

这对齐作者公开 TestSingleQuery 的时钟、调用边界、计数位置和
QPS 换算口径；不声称复现论文绝对数值。硬件、编译、缓存状态和
机器负载仍会不同。尤其当前入口先做计时外正确性核对，因此不能
称为复现原入口的冷缓存状态。原公开代码漏计入口 NDC；这里在 beam
计时后补 1，让所有 NDC 文件统一包含入口，计时函数本身仍保持原状。

原始与批量模式分别存入新目录。manifest 保存 timing_protocol；
跨图比较拒绝混用两种协议。original 额外生成 timings.csv.latencies.csv。
原公开 rderr 实际输出距离比，未减 1；本模式复现该计时外计算，
打印 mean distance ratio，不将它冒充论文定义的 rderr。

## 对照与 recovered/lost 的精确定义

对同一查询 q、同一图 G、同一 ef、同一 K：

- B：beam 最终返回的节点集合。
- T：bound-twohop 最终返回的节点集合。
- GT：精确 top-K 真值集合。
- recovered_gt_total = |(T ∩ GT) \ B|。
- lost_gt_total = |(B ∩ GT) \ T|。
- recall_twohop - recall_beam = (recovered - lost)/K。

两张图各自配对；NGFix 的 recovered/lost 相对 NGFix beam，Bottom
的相对 Bottom beam。不是对另一张图的结果，也不是“baseline 从未
计算过距离”的定义。是否计算/扩展过，需要事件表的 baseline 标记。

例如 GT={1,2,3,4}，B={1,2,8,9}，T={1,3,4,7}：recovered={3,4}，
lost={2}，净新增=1，recall 增加 0.25。

独立等价性测试是另一个程序；测速入口内部还有前 200 查询的参考
核对，以及全体查询的计数/不计数内核核对。诊断重新执行 paired
baseline 和 bound-twohop，诊断时间不拿来作主 QPS。

## DARTH 借鉴范围

DARTH §3.1.1 的三类特征是搜索进展、近邻距离、近邻距离分布：
nstep、ndis、ninserts；firstNN、closestNN、furthestNN；均值、方差、
中位数、25/75 分位数。它们不是图节点社区或局部拓扑特征。

来源：https://arxiv.org/html/2505.19001v1#S3.SS1.SSS1

这里只借鉴特征类别，采用明确适合本实现的定义，不声称复现 DARTH
模型或其全部实现细节。当前不训练模型、不添加新触发规则、不加预算。

## 新增 pre-probe 快照

--trace_search_state 1 与 --event_trace_ef 一起使用，新增
*_probe_state.csv，每个实际探测 mid 一行。以
(query_id,ef,cur_id,mid_id,hop) 与原 mid/jump 表连接。
snapshot 在 mid 距离已计算、mid 被拒绝之后，任何第二跳距离计算
之前获取。不使用 GT、最终总展开次数或未来搜索结果。

- state_step：当前扩展步号，包括入口扩展。
- state_ndc / state_ndc_mid / state_ndc_nn：此时累计完整查询距离，
  含当前 mid 和入口；满足 state_ndc=1+state_ndc_mid+state_ndc_nn。
- state_result_updates：截至此时非删除节点进入 ef 结果堆的次数，含
  初始入口；不等同于 candidate_set 所有 push 次数。
- state_topk_updates：实际接纳节点导致 top-K 填充或严格距离改善的
  次数，含入口；忽略仅等距离的 ID 替换。
- state_steps_since_topk_update：当前步号减最后一次 top-K 更新步号。
- state_discovery_depth：cur 首次被接纳时的发现路径深度，入口为 0，
  普通边 +1，二跳路径 +2。不是图中最短路径距离，也不是 HNSW 层号。
- state_frontier_size / state_beam_size / state_topk_size：候选队列、
  ef 结果堆与可用 top-K 的大小。
- state_entry_dist：本次入口距离，作为 firstNN 的本实现近似定义。
- beam_*：当前 ef 结果堆距离的 min/max/mean/variance/median/q25/q75。
- topk_*：上述堆中距离最小的 min(K,size) 个节点的同样统计；variance
  是总体方差，分位数采用线性插值。

复制有效堆槽并排序，不调用会修改堆的 get_result()。状态观测不
改变 visited、heap、算法接纳条件或距离计算数量。只有独立诊断启用，
两个测速内核不读取这些字段。

## 已有和扩展的结构/位置统计

原 *_jump_events.csv 已记录实际两条边的 first_edge_type、
second_edge_type、EH、cur/mid/nn ID、三者 query 距离，节点原始/
NGFix/RFix 度数、repair_ratio，三组数据距离、邻居交集/Jaccard，
反向边以及 cur->nn 是否有直接边。H=0、N=1、R=2，构成九种边对。

位置用节点 ID、发现路径深度、局部 repair composition 和邻居重合
描述；BASE_ONLY/MIXED/REPAIR_ONLY 表示节点出边组成，不是社区检测。
没有计算全图最短路、全图入度、全图聚类或真实社区标签。
分析器还输出 cur/mid/nn 在不同查询上的复用集中度。

analyze_bound_features.py 输出：

- probe_feature_distributions.csv：失败无新距离、失败有新距离、成功
  无新增 GT、成功补回 GT 的状态/结构特征分布。
- jump_feature_distributions.csv：每类已接纳 jump 的特征分布。
- probe_feature_yield_bins.csv：特征分桶后的接纳率、有效率、查询平衡
  有效率、二跳 NDC、每千次二跳距离对应的归因新增 GT。
- jump_by_edge_pair.csv：HH/HN/HR/NH/NN/NR/RH/RN/RR。
- jump_by_local_region.csv：cur->mid->nn 的局部出边类型组成。
- jump_by_nn_visibility.csv：baseline 未评分/评分未扩展/已扩展 nn。
- node_reuse.csv：各类事件 cur/mid/nn 的复用集中程度。

只看成功 jump 会产生选择偏差；*_jump_mids.csv 及新增快照包含所有
实际探测，包括失败和零新候选探测。没有启动探测的 mid 不在快照中。
topk/beam 特征是在探测前可知；NN 排名、GT/useful、后继贡献、最终
hop_frac 是事后标签/诊断信息，不能作为探测前触发输入。
局部边长/邻居重合是静态图特征，但在线提取或缓存它们有成本，未来
策略实验必须计入。统计关联不证明因果，也不能把 ancestor GT 贡献
相加当作唯一新增 GT。

## 同步与运行

本轮改动文件：

- ngfixlib/graph/hnsw_ngfix.h
- ngfixlib/graph/bound_search_snapshot.h（新增）
- test/bench_hnsw_unbounded_clean.cc
- test/test_unbounded_clean_search.cc
- test/search_hnsw_ngfix.cc
- test/tools/result_evaluation.h
- analysis/run_hnsw_stage1.py
- analysis/analyze_hnsw_stage1.py
- analysis/compare_graph_k_matrix.py
- analysis/analyze_bound_features.py（新增）
- analysis/test_hnsw_stage1.py
- analysis/test_bound_features.py（新增）
- shell/search_webvid_hnsw_stage1.sh
- shell/trace_webvid_bound_features.sh（新增）

已有搜索算法/脚本保留；先前的同步包未自动更新，直接复制这些文件。

```bash
export ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
export DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
cmake --build "$ROOT/build" --target bench_hnsw_unbounded_clean test_unbounded_clean_search search_hnsw_ngfix -j 4
"$ROOT/build/test/test_unbounded_clean_search"

SUITE_OUT="$ROOT/result/graph_k_original_quick_v1" CPU=1 QUERY_LIMIT=1000 \
ROUNDS=1 EF_VALUES=100,200,400 TIMING_PROTOCOL=original RUN_DIAGNOSTICS=0 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

机制单独运行，不重跑 QPS：

```bash
python3 -c 'import pandas,numpy'
OUT="$ROOT/result/bound_features_v1" CPU=1 TRACE_LIMIT=50 TRACE_EF=200 \
bash "$ROOT/shell/trace_webvid_bound_features.sh"
```

如缺少分析依赖，可在服务器安装 pandas/numpy 后执行上述命令。
先 50 条是为了控制时间与文件量；结果支持初步描述，不替代独立
样本复核。先前已经完成的 batch 性能结果仍可用，但不要与 original
协议混成一条曲线。初步特征选取使用诊断查询，验证规则用独立查询。

本地 Windows 无 Linux C++ 编译工具链；新增 C++ 计时/快照路径尚需
服务器编译验证。独立测试已加入开启快照后结果/NDC 不变的检查。
