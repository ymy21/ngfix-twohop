# HNSW Bottom / NGFix × K=10 / K=100

每个图/K 组合都运行 beam 和 bound_unbounded。两张图对同一 K 使用
完全相同的 ef 网格和查询范围；默认 CPU=1、单工作线程、10k 查询、
5 轮随机顺序计时、每点每轮预热一次、批次至少 1 秒。
K=10 使用原 GT 的前 10 项，K=100 使用前 100 项，不需要重算真值。
Recall@10 与 Recall@100 分开评价，不应混成同一个 recall 指标。

## 同步清单

更新已有文件（只添加新搜索内核 include 和新编译目标，旧入口保留）：

- ngfixlib/graph/hnsw_ngfix.h
- test/CMakeLists.txt

新增或覆盖为当前版本：

- ngfixlib/graph/hnsw_ngfix_unbounded_clean_impl.h
- test/bench_hnsw_unbounded_clean.cc
- test/test_unbounded_clean_search.cc
- analysis/verify_base_index.py
- analysis/run_hnsw_stage1.py
- analysis/analyze_hnsw_stage1.py
- analysis/summarize_hnsw_mechanism.py
- analysis/compare_graph_k_matrix.py
- shell/search_webvid_hnsw_stage1.sh
- shell/search_webvid_graph_k_matrix.sh

依赖文件也包含在同步包里，保持原逻辑：

- ngfixlib/graph/hnsw_ngfix_clean32_impl.h
- analysis/run_clean_qps.py

测试和说明文件：

- analysis/test_hnsw_stage1.py
- analysis/HNSW_STAGE1_EXPERIMENT.md
- analysis/GRAPH_K_MATRIX_COMMANDS.md

同步包不包含或覆盖 data/、result/、旧二跳脚本和旧测速实现。
其余现有 NGFix 源文件需与当前工作区的已有版本一致。

## 初始化与编译

将 graph_k_matrix_sync_20261013.zip 上传到
/data/RoarGraph-main/ZHENGXIN/ngfix/，执行：

```bash
set -euo pipefail
export ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
export DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python3 -m zipfile -e "$ROOT/../graph_k_matrix_sync_20261013.zip" "$ROOT"
python3 "$ROOT/analysis/test_hnsw_stage1.py"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target bench_hnsw_unbounded_clean test_unbounded_clean_search search_hnsw_ngfix -j 4
"$ROOT/build/test/test_unbounded_clean_search"
```

不需要 chmod；下面均通过 bash 调用。
matplotlib 可选：没安装时仍输出完整 CSV，安装后重跑汇总即可生成图。

## 四组小规模检查

```bash
SUITE_OUT="$ROOT/result/graph_k_smoke_v1" \
QUERY_LIMIT=200 ROUNDS=1 EF_VALUES=100,200 RUN_DIAGNOSTICS=0 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

小规模检查也覆盖两个 K 和两张图，只用于验证流水线；ef=100/200
使用共同网格。不能把 200 查询、一轮计时作为最终性能证据。

## 完整运行（干净曲线、配对机制统计、事件）

```bash
SUITE_OUT="$ROOT/result/graph_k_full_v1" \
CPU=1 QUERY_LIMIT=10000 ROUNDS=5 RUN_DIAGNOSTICS=1 TRACE_LIMIT=1000 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

按 hnsw_k10、ngfix_k10、hnsw_k100、ngfix_k100 串行运行，避免测速
作业互相竞争。无限二跳可能耗时较长，可以先只跑干净曲线：

```bash
SUITE_OUT="$ROOT/result/graph_k_clean_v1" \
CPU=1 QUERY_LIMIT=10000 ROUNDS=5 RUN_DIAGNOSTICS=0 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

完整运行和只跑曲线是两种选择，不必都执行。脚本拒绝覆盖旧目录；
重新实验请改 v1 为 v2。每个干净测试点在计时外验证参考算法。

## 单独运行某一组合

```bash
GRAPH=hnsw K=10 OUT="$ROOT/result/hnsw_k10_full_v1" bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
GRAPH=ngfix K=10 OUT="$ROOT/result/ngfix_k10_full_v1" bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
GRAPH=hnsw K=100 OUT="$ROOT/result/hnsw_k100_full_v1" bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
GRAPH=ngfix K=100 OUT="$ROOT/result/ngfix_k100_full_v1" bash "$ROOT/shell/search_webvid_hnsw_stage1.sh"
```

这是替代矩阵入口的手动方式，不必再重复执行。矩阵汇总程序期望
四个子目录在同一 SUITE_OUT 下并使用 hnsw_k10 等固定名字。

## 汇总与结果位置

矩阵结束会自动汇总；需要修改 recall 目标或重新画图时：

```bash
python3 "$ROOT/analysis/compare_graph_k_matrix.py" \
  --indir "$ROOT/result/graph_k_full_v1" --targets .90,.95,.97,.98,.99,.995
```

comparison/ 下输出：

- all_recall_qps_ndc_curves.csv：两个 K 各四条曲线。
- k10_at_target_recall.csv、k100_at_target_recall.csv：每个目标上各方法
  最快达到目标的实测点、实际 recall、ef、QPS、NDC 与速度比。
- k10_fixed_hnsw_hard_queries.csv、k100_fixed_hnsw_hard_queries.csv：
  同一 K 固定 HNSW baseline ef=200 下的最难 20%，再比较四种方案。
- all_mechanism_summaries.csv：各图/K 的配对机制汇总（运行诊断时）。
- k10_recall_qps_ndc.png、k100_recall_qps_ndc.png：安装 matplotlib 时生成。

各组合目录下另有 clean/ 原始计时、逐查询 Recall/NDC、manifest，
diagnostics/ 配对统计、事件和 summary/twohop_edge_pair_summary.csv。

修复图机制统计中的“新增 GT”是相对修复图自己的 baseline；原图
统计则相对原图 baseline。两者分母不同，不能直接当作同一批补回
近邻的重合率。共同困难组表用于比较同一批查询。若要比较节点级
恢复重合率，还需另外导出两边最终结果节点 ID。

同 recall 目标的实际 recall 差距大时加密 ef 再复测。诊断 QPS 不
用于性能图，NDC 分位数不代表延迟分位数。索引构建/修复成本暂未测。

本地已经验证 Python 流程；当前 Windows 环境不能编译 Linux C++，
服务器必须完成以上编译与等价性测试。
