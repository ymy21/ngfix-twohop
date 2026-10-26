# 二跳搜索实验归档：在 NGFix 上的实现、实验与阶段结论

归档日期：2026-10-26。状态：停止继续推进当前二跳实验路线，保留代码、配置与已有结果，供后续技术整理使用。

本文根据本地代码、日志、CSV 和已有复盘整理，没有重新运行服务器实验。源码工作区包含未提交修改；本地 Git HEAD 为 `9acd987cf1c420069f8fd602c0d4e4aa81053558`，该哈希不包含这些修改，最终归档版本以服务器提交为准。

## 1. 研究问题与基础实现

基于 NGFix 公开实现，探索：在不新增持久修复边的条件下，能否让搜索越过被 beam 拒绝的中间节点，以有限二跳找回普通搜索漏掉的近邻，并在相近召回率下降低完整距离计算或提高 QPS。

核心动作是 `cur → mid → nn`：cur 正常展开，mid 已计算与 query 的距离但被当前 beam 条件拒绝；额外访问 mid 的出邻居，对 nn 评分并按实时 bound 决定是否进入正常搜索。mid 不因这次例外探测而自动进入正常队列。不同版本在探测时机、mid 筛选、nn 选择及预算上不同。

上游地址：<https://github.com/yuhuifishash/NGFix>。原始 NGFix/RFix 修图、插入删除和 RaBitQ 入口属于上游功能；文件保留或脚本路径修改不等于本项目重新验证了这些功能的全部效果。本次主要新增的是搜索、诊断和实验管理。

## 2. 数据、图与比较口径

| 项目 | 已核对内容 |
|---|---|
| 主要数据 | CLIP-WebVid，base 2,505,000 个节点，512 维 float |
| 距离 | `ip_float`，实现距离为 `1-dot(q,x)`；已有范数审计，不能直接套用欧式安全界 |
| 原始图 | `_HNSWBottom_M16_efC2000_MEX48.index`，51,379,614 条基础有向边，修复边为 0，平均出度 20.5108 |
| 原始图入口 | 1,103,413；M32 等其他图使用各自文件内入口 |
| 修复图 | `_HNSWNgfix200M16_efC2000_MEX48.index`；已有配对报告记录 38,914,955 条修复边，具体来源类型按对应日志判断 |
| K | 主线 K=100，另有 K=10 消融 |
| 查询范围 | 早期有 20/200 条开发或诊断；完整对照主要是旧 test 前 1,000 条；后续 cur/mid 实验使用 ID 1000–1999；A/B 使用另行拆分的 development 1,000 条 |
| CPU | 主要固定逻辑 CPU1，查询串行单线程 |
| 指标 | Recall@K、完整距离计算次数 NDC、QPS；部分实验保存真实逐查询延迟和 P95/P99 |

注意：这里大量实验比较的是加载后的固定底层图搜索，不代表完整多层 HNSW 系统复现。MEX48 表示修复容量相关配置，不能仅凭文件名判断已经存在修复边。

判定效率时允许两种方法取不同 ef，以实际达到的召回和成本比较。相同 ef 的召回提升、只减少一跳 NDC，均不能单独证明净收益。不同 query 子集、不同计时协议或诊断 QPS 不混用。

## 3. 已做过的主要尝试

### 3.1 普通 beam、DABS 与无约束二跳基线

- 保留普通 beam，扫描 ef，作为所有额外探索的成本对照。
- 对 DABS 做 gamma 网格及 fixed-ef 的 Recall–NDC 对比。早期 200-query 报告中，达到 95% 目标的实测点 NDC 从 5,405.30 降到 5,222.00，约省 3.39%；98%/99% 目标未显示相同优势。这只是开发网格结果。
- 实现即时 `immediate` 二跳：拒绝 mid 后立即扫描其全部未访问 nn，精算、排序并按实时 bound 接纳。关闭额外预算或门禁，不等于无视 visited、beam 剪枝或终止条件。
- 在 Bottom/NGFix 两张图、K=10/100 上按各自 ef 网格比较，共 144 个性能点，其中 72 个二跳点。

已测结论：在 54 个能找到同图、同 K、召回不低的 beam 点的二跳配置上，beam 全部更快；另外 18 个二跳点超出该图 beam 已测召回范围，不能外推裁定。原图二跳的 36 个点均有召回不低且更快的 NGFix beam 点，但没有据此比较整个系统的构建维护成本。

证据：`analysis/HNSW_STAGE1_EXPERIMENT.md`、`result/webvid_stage1/stage1_g000_0020_dense_q200_report.md`、`result/complete_experiment_report_20261013/完整报告.md`、`result/graph_k_original_ngfixgrid_1k_v1/`。

### 3.2 二跳贡献、失败探测与搜索状态诊断

在原搜索上加入独立诊断，记录入口/一跳/二跳 NDC、probe、nn 入队、visited 重复扫描、recovered/lost GT，以及直接找回与后续路由贡献。

对实际二跳记录两条边的 HNSW/NGFix/RFix 类型，形成 HH/HN/HR/NH/NN/NR/RH/RN/RR 分类；记录 EH、cur/mid/nn 距离、节点出边组成、邻接交集、反向边等。探测前保存搜索进度、bound、堆统计、近期 top-K 更新等状态，用于分析触发收益与成本。

机制明细覆盖 8 组固定配置，报告读取约 1,214.8 万条 probe、323.8 万条已接纳 jump。观察到不少二跳距离花在没有任何 nn 入队的 mid 上；有效候选还可能通过后续非 GT 路由点产生收益，因此不能只用“nn 是否在 GT”判断有用。

诊断程序包含来源追踪和状态采集，其时间不作为主性能 QPS。probe 至少接纳一个候选也不等于最终增加 recall。

证据：`analysis/BOUND_FEATURES_AND_TIMING.md`、`result/bound_features_selected_1k_v1/`、`result/complete_experiment_report_20261013/`。

### 3.3 触发门禁、探测调度与预算

围绕“何时值得跳、选择哪个 mid”尝试了：

- mid 到当前 bound 的 gap 门禁、cur 质量/搜索深度等门禁及其组合；
- `immediate` 与先完成普通一跳再探测的 `after_onehop` 调度，以及窗口调度；
- 按 gap 等规则选择少量 mid；每 cur、每 mid、每 query 的二跳精算预算；
- 保留树模型/策略训练与运行工具，作为可继续研究的接口；不能仅凭训练脚本存在声称模型已获验证。

早期 200-query 实测中，gap 等规则能相对原无约束二跳节省 NDC，但 `selection_dev200` 的 14/14 非 baseline 点被普通 beam 在 Recall–NDC 上支配，`trigger_gap_dev200` 为 12/14。其余未被支配也不等于胜出。上述扫描带诊断，不能以其时间宣称部署加速。

证据：`shell/search_webvid_jump_gates.sh`、`shell/search_webvid_jump_schedule.sh`、`ngfixlib/graph/jump_policy.h`、`analysis/train_jump_policy.py`、`result/jump_rules_core_dev200_v1/`、`result/selection_dev200_20261008_v1/`、`result/trigger_gap_dev200_20261008_v2/`。

### 3.4 第二跳 nn 选择与便宜评分

围绕“选中 mid 后是否必须精算全部 nn”做了 FULL、物理邻接前缀 8 个、随机 8 个（多个 seed）、cheap score 选 8 个的对照；cheap score 使用 16/32 维配置，另有全量精算池上的离线距离排名参考。

诊断与线上搜索分开：完整候选池用于评估有效 nn 保留率、最近邻保留率；实际搜索另行测 Recall–NDC。只在未截断、全部精算的池上计算可靠的离线排名标签。离线 oracle 排序已经使用完整距离，不能当作免费可部署筛选。

保留了 20-query 诊断与 200-query 在线原始结果；本归档不把这些开发记录概括为已验证的最终提速方法。

证据：`ngfixlib/graph/jump_nn_selection.h`、`analysis/run_jump_nn_selection.py`、`analysis/analyze_jump_nn_selection.py`、`result/nn_diag20_d16_20261008_v1/`、`result/nn_diag20_d32_20261008_v1/`、`result/nn_online200_d16_d32_20261008_v2/`。

### 3.5 选择性前缀二跳及 cur/mid FULL

实现过独立 selective-prefix 版本：正常一跳完成后，用 cur slack、mid gap 过滤，最多选 2 个 mid，各取前 8 个未访问且互异的 nn，每 query 最多额外精算 1,024 次。前缀按物理邻接顺序，不是 query 最近的 8 个。该版本有实现、测试与运行说明，不将说明中的默认全网格自动视为已完成实测。

随后重点测了“只筛 cur/mid，选中 mid 后 FULL 精算全部新邻居”：

```text
b = 完成当前 cur 普通一跳后的实时 beam bound
s = max(abs(b), 1e-6)
mid_gap = (d(q,mid) - b) / s
cur_slack = (b - d(q,cur)) / s
```

配置包括 `mid_gap ≤ 0.0266`、再加 `cur_slack ≥ 0.0127` 的 `cur_mid`、`gap/slack ≤ 1` 的 `ratio_1`，以及可选 ratio_2/cur_only/unfiltered。阈值来自旧特征分桶，属于探索起点。候选 mid 按 query 距离排序，每次实际探测前用最新 bound 重检。

M16 的独立 1,000-query、6 个 ef、两轮结果确认：筛选显著压低无约束二跳成本，但还未证明优于扩大 beam。例如 `cur_mid ef800` 为 Recall 93.796%、QPS 149.37、NDC 21,477.18；同批 `immediate FULL ef400` 为 93.585%、QPS 75.77、NDC 45,388.24，说明能改进旧二跳。另一方面，`beam ef800` 的 91.079%/278.49/10,110.13 同时优于 `cur_mid ef400` 的 89.645%/269.25/12,156.60。

原 FULL 与新筛选还改变了调度时机，不能把全部收益归因于门槛。该轮 beam 最高只到 ef800，不能据此判断更高召回的全部胜负。

另有 M32 底层图与 M16 的筛选运行记录，用于观察增大基础度数的替代效果；它不是严格等内存或完整多层 HNSW 对照，不在此新增统一胜负结论。

证据：`analysis/HNSW选择性二跳运行说明_20261015.md`、`analysis/HNSW_cur_mid全邻居实验说明_20261015.md`、`result/m16/report_20261015/report.md`、`result/m16/gates/`、`result/m32/gates/`。

### 3.6 NGFix 残余漏检与补边利用

既比较原图二跳，也在修复后的 NGFix 图上加二跳，检验其对剩余漏检的补救。

旧 test 前 1,000 条、K100/ef100：NGFix beam Recall 92.290%，NGFix 二跳 98.526%；recovered 6,341 个 GT、lost 105 个，净补回原漏检约 80.88%。这是相同 ef 下的补救能力，同时有很高额外成本，不能当作同成本优势。

按 NGFix 困难 query 固定分组后，原图二跳存在局部召回互补，但扩大 NGFix beam ef 可消除已观察的若干优势；没有证据表明它一般性地高效解决了 NGFix 最难的查询。

补边利用分析进一步核对旧 10k query：ef100 时 NGFix 边贡献约 65.76% 的邻居距离计算、63.83% 的首次发现来源展开。普通 beam 已经通过邻接遍历使用补边，不能说“不读历史 EH 就没有用到修复”。首次发现来源也不是该边不可替代的因果证明。

证据：`result/residual_query_analysis_20261014/残余难查询验证与改进方案.md`、`result/ngfix_beam_theory_20261021/NGFix补边利用与纯二跳可行性.md`。

### 3.7 基于历史 query 的 A/B 边标记

这是后期在固定、未修复 Bottom 图上的主要验证。对历史 q 令 S 为精确 top-200；考虑已有边 `c→m`，要求 c∈S、m∉S：

- A：存在 `m→n`，n∈S，n≠c，且 n 不是 c 的直接出邻居。
- B：满足 A，且至少一个这样的 n 无法通过 S 内的有向路径从 c 到达。因此 B⊆A。
- 对所有训练 query 做 union：某条边出现一次正例即永久标记。线上普通一跳完成后，只对相应标记允许的被拒 mid 做 FULL 二跳。

离线标记读取 GT ID 和图邻接，不直接使用 query 向量或实际搜索状态；不保存每次正例对应的出口 nn，不能称为已训练的 query 条件收益模型。

使用完整训练 workload 的随机互斥行拆分，seed=17；训练量 2,505/12,525/25,050 为嵌套前缀，另取 development/calibration 各 1,000 条。三档比例以 base 2,505,000 为分母。GT500 已做抽样穷举 IP 核对，标记使用前 200 列。行 ID 互斥不证明消除了语义近重复。

已完成主结果为 development 1,000 条、K100、8 个 ef、3 轮计时。共有 9 方法×8 ef=72 个不同配置点（包含一个 cur/mid 几何筛选对照）；重复执行的 Beam 不另算独立方法。本批没有 calibration 在线性能或独立 10k test 结果。

下面摘录同 ef800 的部分实测点，QPS 为三轮中位数；相同 ef 不代表相同 recall。

| 方法 | 标记训练 query 数 | Recall@100 | 平均 NDC | QPS |
|---|---:|---:|---:|---:|
| Beam | 0 | 91.369% | 10,034.971 | 277.194 |
| FULL | 0 | 96.376% | 78,084.693 | 41.305 |
| A | 2,505 | 93.395% | 21,525.065 | 145.425 |
| B | 2,505 | 92.956% | 17,244.567 | 181.752 |
| A | 25,050 | 95.230% | 41,911.897 | 75.616 |
| B | 25,050 | 94.874% | 35,048.218 | 90.577 |

相近召回的直接对照：Beam ef2400 为 96.284% / NDC 26,821.461 / QPS 101.965；B-25,050 ef1200 为 96.265% / NDC 46,748.683 / QPS 67.271。B 的 NDC 高约 74.30%，QPS 低约 34.03%。

48 个 A/B 点中，36 个处于已测 Beam 最大召回 97.033% 的范围内，全部存在召回不低、NDC 更少的 Beam 点；其中 32 个还被同一 Beam 点在 QPS 上超过。另 12 个超出 Beam 已测召回范围，不下外推结论。

扩大历史 query 提高了覆盖，也放大在线成本：最大训练档 A/B 标记约占全图边 17.546%/8.952%；ef800 的 B 每 query 展开 mid 从约 460.6 增到 1,688.3，至少一个 nn 入队的展开比例从 23.05% 降到 17.64%。标记能压缩盲目 FULL，但尚未形成相对扩大 beam 的计算优势。

证据：`ngfixlib/graph/ab_marks.h`、`test/build_ab_marks.cc`、`analysis/ab_stage1.py`、`analysis/WEBVID_AB_REVIEW_20261022.md`、`result/webvid_ab_s17_L200/`。

## 4. EH 相关分析与尚未完成的方向

源码解读核对的 EH 是所有有向路径的最小排名瓶颈：

```text
EH_q(c,n) = min_{P:c→n} max_{x∈P} r_q(x)
```

不是最短跳数路径，也不是固定三元组的最大 rank。本地实现使用 rank 前缀与传递闭包，存在计算窗口截断；EH_INF 不等于全图永久不可达。历史 EH 不是当前 query 的在线收益分数，A/B 布尔标签也不等同完整 EH。

做过固定轨迹的免费距离折扣：旧 test 前 1,000 条、K100/ef400 的原图 FULL NDC 约 44,775；免费去掉无 nn 入队 mid 的二跳计算后约 8,991，进一步只算入队 nn 后约 4,107。相同 query 的实际 NGFix beam ef130 为 Recall 94.266%、NDC 4,937；原 FULL recall 为 93.970%。这些折扣保留旧轨迹，没有真正删操作重跑，也没有预测开销，不能作为可部署 oracle 成绩或理论下界。

更大 EH 窗口、EH/状态条件门控、保存多出口 nn、真正重跑的 oracle、原图二跳对修复边的结构覆盖、互补多入口等，已有分析或设计文件。它们没有在本归档中被认定为完成了最终有效性验证；多入口不属于本次已完成二跳实测成果。

证据：`result/eh_jump_headroom_20261020/`、`result/eh_state_gate_20261020/`、`result/three_routes_20261020/`、`analysis/WEBVID_GATE_MULTI_ENTRY_PLAN_20261021.md`。

## 5. 在 NGFix 上保留下来的代码资产

| 位置 | 内容 |
|---|---|
| `ngfixlib/graph/hnsw_ngfix.h` | 搜索扩展、诊断与新实现接入；与上游差异以 Git diff 为准 |
| `bound_twohop_gate.h`、`bound_search_snapshot.h` | 门禁与探测前状态支持 |
| `jump_policy.h`、`jump_nn_selection.h` | 策略、调度与 nn 选择支持 |
| `hnsw_ngfix_unbounded_clean_impl.h` | 无约束二跳独立测速实现 |
| `hnsw_ngfix_clean32_impl.h` | 便宜评分相关独立测速实现 |
| `hnsw_ngfix_selective_prefix_impl.h` | 有预算的前缀二跳 |
| `hnsw_ngfix_cur_mid_full_impl.h` | cur/mid 门禁、FULL nn 及 A/B 标记接入 |
| `ab_marks.h`、`test/build_ab_marks.cc` | 离线 A/B 标记构建、存取与统计 |
| `test/search_hnsw_ngfix.cc`、`test/bench_*.cc` | 诊断和独立性能入口 |
| `test/test_*.cc`、`analysis/test_*.py` | 算法等价、边界、工作流及汇总校验 |
| `shell/`、`analysis/` | ef 网格、批处理、输入身份核对、GT 核对、统计、图表与复盘 |

`ngfixlib/metric/`、`rabitqlib/fastscan/` 等还存在与上游不同的代码，属于归档范围；不要仅据修改状态将其描述为已证明有效的新算法。

原始与独立内核通过参考实现、计数/不计数结果核对来保持口径。`original` 协议逐查询计时且计入 NDC；`batch` 协议批量计时、关闭机制和 NDC 统计。报告使用各轮实际协议，不能跨协议拼接速度结论。

## 6. 停止推进的阶段判断

二跳的补救能力已经观察到，门禁和历史标记也能减少盲目扩展；当前瓶颈是发现出口的精算成本，以及静态历史结构与当前 query/搜索状态收益不匹配。选中 mid 后 FULL nn 仍然昂贵，增大训练量并未自动解决这一点。

因此冻结当前代码和结果，停止继续扩大本路线实验。现有材料适合保留为技术方案、实现过程和实测边界的记录；尚未验证的性能优势、专利新颖性或可授权性，不在本归档中作结论。

## 7. 归档和复现材料

- GitHub 保存源码、实验脚本、本说明，以及挑选的小型报告与 CSV。
- 完整 `data/`、索引、训练标记二进制和大型 `result/`/`output/` 留在服务器或独立备份；只存源码不能完整复现实验。
- 保留对应 `manifest.json`/`run.json`、`commands.txt`、输入哈希、query 拆分映射、原始计时 CSV 和日志。部分归档只记录路径，迁移时需要一并维护这些对应关系。
- 旧大目录与中断运行不混合认定为完成结果；复现时使用新输出目录，避免覆盖冻结数据。
- 按 [GITHUB_UPLOAD.md](GITHUB_UPLOAD.md) 从服务器创建归档提交，并把本说明一同上传。已有 `README.md` 保留上游项目说明。
