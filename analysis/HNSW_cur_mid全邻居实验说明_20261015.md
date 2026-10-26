# 本轮实验：只筛 cur/mid，选中的 mid 精算全部新邻居

本说明对应 `shell/search_webvid_cur_mid_full.sh`，替代上一版 prefix 运行方案作为本轮推荐。旧的无约束二跳、NGFix 二跳和 selective-prefix 实现均保留；运行本说明的新脚本即可。

补充：主实验 INDEX 现在允许传入 M16 或 M32 未修复图，因此可通过 INDEX 指定M32，同时设置 RUN_M32=0、RUN_NGFIX=0，使所有选定筛选规则直接运行在M32上。原始immediate无约束二跳仍使用 run_hnsw_stage1.py/bench_hnsw_unbounded_clean 单独运行；新规则使用after_onehop，不能把关闭规则的新入口当成原始immediate二跳。

## 规则及含义

距离都是 query 到节点的距离，越小越好。不是 cur 与 mid 之间的边长。令 b 为完成当前 cur 的普通一跳之后，候选队列的当前 bound，s=max(abs(b),1e-6)：

- mid gap = (d(q,mid)-b)/s。
- cur slack = (b-d(q,cur))/s。
- 比例规则 r=(d(q,mid)-b)/(b-d(q,cur))，仅在分母>0时允许。

只考虑当前 cur 的一跳中已精算、被 beam 拒绝的 mid。先完成全部普通一跳，再按 mid 的 query 距离从小到大处理通过规则的 mid。每次实际 probe 前重新用最新 b 检查规则；先前 probe 收紧 b 可能导致后续 mid 不再通过。此前已通过几何筛选但未被数量选择的 mid 计入 blocked_selection；重新检查失败则计入对应 gate 拦截。

| profile | mid 要求 | cur 要求 | 其他要求 |
|---|---|---|---|
| mid_gap | gap <= 0.0266 | 无 | 无 |
| cur_mid | gap <= 0.0266 | slack >= 0.0127 | 无 |
| ratio_1 | 无独立 gap 阈值 | slack >= 0.0127 | gap/slack <= 1 |
| ratio_2（可选） | 无独立 gap 阈值 | slack >= 0.0127 | gap/slack <= 2 |
| cur_only（可选） | 无 | slack >= 0.0127 | 无 |
| unfiltered（可选） | 无 | 无 | 无 |

默认 MID_CAP=0，意味着不按数量截断 mid，只按上述 cur/mid 规则筛选。如果另设 MID_CAP=1/2，则每个 cur 仅选最靠近 bound 的前 1/2 个合格 mid，这是另一项消融，不能与默认结果混称。

对每个实际 probe：扫描 mid 的全部邻接项，跳过 visited，精算全部新节点，按精算距离排序，再以实时 beam 条件接纳。没有 NN prefix、NN 抽样、NN cheap 预筛选，也没有 query/per-cur/per-mid 二跳精算预算。所有精算节点及时标 visited；重复路径仍可扫描同一节点，但不会反复精算。

建议以 cur_mid 为主方法：两个阈值独立，便于说明与消融；ratio_1 作比较。比例不是“选一定百分比的 mid”，也不是“抽一定比例的 NN”。按排名取固定百分比会在没有合适 mid 时也强行选出节点，不建议用作本轮主规则。

0.0266、0.0127 来自此前 HNSW K10 ef100 的分桶边界（近边界 gap 的第一分位、cur slack 的中位数）。它们只作为探索起点，不是已验证最优值；此前是 immediate 时刻，本轮是 after-onehop 且 live recheck，分布会改变。归一化阈值比原始距离常数更便于比较不同 bound，但仍依赖距离定义；IP 上不是几何安全界。大的 cur slack 表示 cur 相对当前 beam 较好，不能直接据此断言存在图缺陷。

## M32 对照的边界

论文第6.1节写普通 HNSW 使用 M=32、efC=2000；跨模态 NGFix 基础图使用 M=16、efC=2000、MEX=48，并只保留底层。
来源：https://arxiv.org/html/2510.22316v1#S6.SS1

这里用现有 `build_hnsw_bottom` 补建 M=32、M0=64、efC=2000、MEX=48 的**未修复底层图**。MEX 保留为一致的格式/可扩展容量配置，不等于实际加入48条修复边。构图仍是原实现的32线程，仅输入数据库，不使用历史查询。新图使用自身保存的入口，不能假定仍是1103413。不同图可能有构图随机性/并行顺序差异。

这是增加底层度数的系统对照，不能写成完整复现论文的多层 HNSW 基线。M16与其NGFix修复图仍是同一基础图的机制比较。M32最大底层度64；M16修复图允许基础32+修复48，不是严格等内存。以日志实际平均出度及 manifest 的文件字节数报告成本。脚本在每个配置计时前调用 printGraphInfo，记录平均出度、容量、入口及边类型计数。

## 本次新增/修改文件

在本地 `C:\Users\姚明奕\Desktop\code\10.12\NGFix` 下，按同名相对路径复制到服务器 `$ROOT`：

1. `ngfixlib/graph/hnsw_ngfix.h`（增加新实现 include）
2. `ngfixlib/graph/hnsw_ngfix_cur_mid_full_impl.h`（新算法）
3. `test/CMakeLists.txt`（增加两个编译目标）
4. `test/bench_hnsw_cur_mid_full.cc`（性能入口）
5. `test/test_cur_mid_full_search.cc`（等价性测试）
6. `analysis/run_hnsw_cur_mid_full.py`（实验调度、输入记录）
7. `analysis/analyze_hnsw_cur_mid_full.py`（结果校验与汇总）
8. `analysis/test_hnsw_cur_mid_full.py`（汇总/调度测试）
9. `shell/search_webvid_cur_mid_full.sh`（运行脚本）
10. `shell/build_webvid_hnsw_m32.sh`（构图脚本）

还依赖此前已有 `shell/webvid_ngfix_ef_grid.sh`。如果上一版 selective-prefix 文件尚未复制到服务器，务必同时复制以下三个文件，否则当前总头文件/CMake引用它们会导致配置或编译失败：

- `ngfixlib/graph/hnsw_ngfix_selective_prefix_impl.h`
- `test/bench_hnsw_selective_prefix.cc`
- `test/test_selective_prefix_search.cc`

此前的 clean32、unbounded、jump gate/NN selection 等文件也必须已是本地对应版本。无需复制 data、result、output，也无需删除之前结果。本次没有修改 `build_hnsw_bottom.cc`；构图调用原有入口。

## 服务器命令

复制后先编译与测试。下面 DATA 沿用已有 WebVid 脚本路径，若服务器实际路径不同，修改 DATA；文件名可分别用 BASE/QUERY/GT 覆盖。

```bash
ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
export ROOT DATA

python3 - "$ROOT" <<'PY'
from pathlib import Path
import sys
r=Path(sys.argv[1])
for name in ('shell/search_webvid_cur_mid_full.sh','shell/build_webvid_hnsw_m32.sh'):
    p=r/name
    p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
PY

cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target bench_hnsw_cur_mid_full test_cur_mid_full_search build_hnsw_bottom -j4
"$ROOT/build/test/test_cur_mid_full_search"
python3 "$ROOT/analysis/test_hnsw_cur_mid_full.py"
```

本地缺少Linux C++编译环境，未声称C++测试已运行；上面的测试必须在服务器通过后再跑正式实验。C++测试与已有带统计参考入口比较完整返回结果和NDC，覆盖L2/IP、K10/K100、gate/ratio、mid数量不限和限额、重复邻居、删除节点、visited复用、实时bound重检查；另用单个probe精算20个NN的用例检查没有残留prefix=8。

先在原M16图上快速检查新算法，不等M32构建：

```bash
SUITE_OUT="$ROOT/result/cur_mid_full_smoke_20261015_v1" \
KS=10 EF_VALUES=20,100 QUERY_LIMIT=50 VERIFY_LIMIT=50 ROUNDS=1 \
RUN_M32=0 RUN_NGFIX=0 CPU=1 \
bash "$ROOT/shell/search_webvid_cur_mid_full.sh"
```

构建M32，只需一次；如果已经有这个文件，先检查是否为完整的同配置图，直接复用，脚本拒绝覆盖：

```bash
bash "$ROOT/shell/build_webvid_hnsw_m32.sh"
```

这是250.5万向量、efC2000的重新构图，可能是整轮新增工作中最耗时的部分；不要因为查询只用1k而误认为构图也只用1k。构图日志保存为索引名加 `.build.log`，中间文件后缀 `.building`。构图中止留下该文件时需先检查，再决定如何处理，脚本不自动删除。

正式实验：

```bash
SUITE_OUT="$ROOT/result/cur_mid_full_m32_1k_v1" \
CPU=1 QUERY_OFFSET=1000 QUERY_LIMIT=1000 VERIFY_LIMIT=1000 ROUNDS=2 \
MID_CAP=0 MID_GAP=0.0266 CUR_SLACK=0.0127 \
PROFILES=mid_gap,cur_mid,ratio_1 \
RUN_M32=1 RUN_NGFIX=1 \
bash "$ROOT/shell/search_webvid_cur_mid_full.sh"
```

两种K都使用同一批 query IDs 1000–1999（0-based），避免直接复用此前挑阈值的前1k查询。按反馈继续调参后，这批也属于验证集；最终报告需要保留新的独立测试查询。

默认ef沿用已有NGFix网格：

- K10：10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000。
- K100：100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000。

M16每个profile分别运行同图beam与cur_mid_full；M32和NGFix分别只跑beam，避免额外扩展实验范围。M16重复beam用于观察运行波动，汇总以cur_mid目录的beam作共同锚点。NGFix二跳旧实验保留且不由本脚本重跑。

只想先看较短一轮，可另用新目录并在命令中增加：

```bash
EF_VALUES_K10=20,50,100,150,200,400
EF_VALUES_K100=100,150,200,300,400,800
```

上述两个赋值需要像其他参数一样放在同一次 bash 调用前（带续行），或先 export。已核对共享脚本，覆盖变量确实是 EF_VALUES_K10/EF_VALUES_K100。

可选进一步消融：PROFILES=unfiltered,cur_only,mid_gap,cur_mid,ratio_1,ratio_2；仍MID_CAP=0，仍全部NN精算。unfiltered是after-onehop+gap排序、无几何限制，和旧immediate无约束二跳不完全相同，应分开标记。它是隔离调度改变与几何规则收益的必要对照，但成本可能高，可先在少量ef上做。

## 计时、统计与比较

单查询单线程，taskset绑定逻辑CPU1（不是“CPU数量=1”的参数）。使用original协议：gettimeofday紧贴一次搜索调用前后，baseline直接调用公开 searchKnn，二跳调用去掉诊断映射的等价实现，保留NDC计数。GT评估、文件输出、特征计数和参考等价检查均在计时之外。搜索分配、选择mid、邻居扫描、距离计算、排序、入队的时间都算在内。每轮QPS=N/sum(search_seconds)，两轮报告QPS中位数及min/max；P95/P99用两轮单查询时延汇总。完整实验总墙钟时间大于计时之和。

未计时阶段对全部1k查询比较无计数/有计数新入口结果，并按 VERIFY_LIMIT 与原带统计参考入口核对完整结果及NDC。默认 VERIFY_LIMIT=1000。baseline原NDC漏计入口，输出在计时后加1统一口径；GT距离评估不计入搜索NDC。

每K目录保存manifest.json、commands.txt，以及每case的benchmark.log、timings.csv、timings.csv.queries.csv、timings.csv.latencies.csv。manifest记录索引大小/头、修改时间、二进制hash、阈值与命令；不是全索引内容hash。

自动汇总在 `k10/analysis` 和 `k100/analysis`：

- points.csv：每个图/规则/ef的recall、平均NDC、QPS中位/min/max、平均/P50/P95/P99时延。
- target_recall.csv：每个方法在recall至少达到0.90/0.95/0.97/0.98/0.99的测量点中取最高QPS，比较M16 beam，同时包含M32 beam、NGFix beam。不是插值后的严格等recall，达到的recall差>0.001会标记needs_denser_ef_grid。
- mechanism.csv：平均二跳NDC、probe、拒绝mid、各gate拦截数、接纳NN数、扫描数、展开数、successful_probe_rate、probe_per_rejected_mid、nn_ndc_share、visited_scan_fraction。

统计口径：

- NDC=入口1次+普通一跳新节点精算+probe中新节点精算。
- rejected_mids：普通一跳精算后当时被beam拒绝的节点次数。
- probes：实际扫描了邻接表的mid次数（即使没有新节点）。
- blocked_cur/gap/ratio：按cur→gap→ratio的顺序第一次失败的拦截原因，含probe前live重检查；不是彼此独立事件。
- blocked_selection：通过几何筛选但未被mid数量上限选中；MID_CAP=0时应为0。
- 每query应满足 rejected_mids=blocked_cur+blocked_gap+blocked_ratio+blocked_selection+probes。
- successful_probes：至少有一个NN进入beam的probe；不是恢复真近邻的probe。
- scanned_nn：probe扫描的邻接项总数，包含visited/重复项；ndc_nn仅计算新精算。
- visited_scan_fraction=1-sum(ndc_nn)/sum(scanned_nn)。visited可能来自一跳、之前probe或重复邻接项。
- nn_ndc_share=sum(ndc_nn)/sum(total_ndc)。

本轮是性能及gate成本消融，不输出旧机制实验的每个probe特征、GT归因链或direct/routing recovered GT；这些旧诊断入口仍保留。不要用successful_probe_rate代替useful probe比例。先看同目标recall是否比增大ef快，再看near-boundary与cur slack是否减少probe/NDC且维持recall。若没有收益，应保留负结果，继续研究触发依据，而不是将门槛描述为已识别局部图障碍。
