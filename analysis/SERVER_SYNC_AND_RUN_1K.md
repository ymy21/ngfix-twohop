# 最新服务器同步与运行清单

本地根目录：C:/Users/姚明奕/Desktop/code/10.12/NGFix

服务器根目录：/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix

将 latest_server_sync_files.txt 中的 22 个文件按相对路径复制到服务器。
清单包含本次整个流程需要的代码和依赖，不要求使用之前的 ZIP 包。
不复制本地 data/、result/、output/ 或 Windows 编译结果。服务器已有
的索引和查询数据继续使用。原有 NGFix 二跳代码和脚本保留。

中止的 result/graph_k_full_v1/ 无需删除，新实验使用两个全新目录。
中止目录中可能包含完整和未完成的子运行，不自动合并为本轮结果。
仅在确实需要释放空间时再自行处理旧结果目录。

下面命令在旧实验已停止、代码复制完成后，在服务器同一个 Bash
终端按顺序执行。请保持其他测速/诊断作业停止，避免相互竞争。

## 1. 初始化、依赖检查、编译、测试

```bash
set -euo pipefail

export ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
export DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 使用 NGFix 公开代码的完整 K-specific ef 网格，清除旧临时覆盖。
unset EF_VALUES EF_VALUES_K10 EF_VALUES_K100 TRACE_EF TRACE_EFS INDEX

python3 -c 'import numpy,pandas' || python3 -m pip install --user numpy pandas

python3 "$ROOT/analysis/test_hnsw_stage1.py"
python3 "$ROOT/analysis/test_bound_features.py"

cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" \
  --target bench_hnsw_unbounded_clean test_unbounded_clean_search search_hnsw_ngfix \
  -j 4

"$ROOT/build/test/test_unbounded_clean_search"
```

matplotlib 可选，未安装仍有完整 CSV；安装后重新汇总可生成图。
不需 chmod；所有 shell 入口通过 bash 调用。

## 2. 性能实验

```bash
SUITE_OUT="$ROOT/result/graph_k_original_ngfixgrid_1k_v1" \
CPU=1 QUERY_LIMIT=1000 ROUNDS=1 \
TIMING_PROTOCOL=original RUN_DIAGNOSTICS=0 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

2 张图、K=10/100、beam/unbounded-bound-twohop，共 144 个测试点。
逐查询 gettimeofday 计时，NDC 在计时内，计时外 GT 评估；QPS 为总
查询数除以总搜索延迟。前 200 查询参考核对保留，全体 1k 查询核对
计数/无计数内核结果。单轮用于初步性能判断，不作为稳定性证据。

结果自动汇总到：

result/graph_k_original_ngfixgrid_1k_v1/comparison/

## 3. 独立机制实验

```bash
OUT="$ROOT/result/bound_features_ngfixgrid_1k_v1" \
CPU=1 TRACE_LIMIT=1000 \
bash "$ROOT/shell/trace_webvid_bound_features.sh"
```

与性能实验相同的前 1k 查询和 ef 网格，72 组同图/同 K/同 ef 的
beam/二跳配对诊断。诊断时间不当作性能 QPS。每个 ef 单独保存
trace.csv、jump_events、jump_mids、probe_state 及 summary/features。
完整事件量大于以前 50 查询单 ef 的配置；trace_plan.log 记录进度。

示例目录：

result/bound_features_ngfixgrid_1k_v1/ngfix_k100/ef200/

## 4. 手动重新汇总（结束后自动完成，通常不必再跑）

```bash
python3 "$ROOT/analysis/compare_graph_k_matrix.py" \
  --indir "$ROOT/result/graph_k_original_ngfixgrid_1k_v1" \
  --targets .90,.95,.97,.98,.99,.995
```

## ef 网格

K=10，20 个：
10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000

K=100，16 个：
100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000

两个新输出目录若已存在，将命令中的 v1 换成 v2，并同步调整手动
汇总路径。脚本不会覆盖已有输出。当前服务器应保留旧中止结果，
除非用户另行决定清理；上述命令不删除任何文件。
