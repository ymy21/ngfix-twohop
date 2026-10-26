# 两个独立实验：1k 查询与 NGFix 公开 ef 网格

性能实验和机制实验均使用 query.10k.fbin/gt.10k.ibin 的前 1000 条，
query_offset=0，CPU=1，IP float，HNSW Bottom/NGFix，K=10/100。
不改图结构、不改搜索策略或二跳预算。

官方来源：https://github.com/yuhuifishash/NGFix/blob/main/test/tools/result_evaluation.h

K=10 的 20 个 ef：
10,15,20,30,40,50,60,70,80,90,100,120,150,180,200,300,400,500,800,1000

K=100 的 16 个 ef：
100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000

本轮更新 shell/search_webvid_hnsw_stage1.sh、
shell/trace_webvid_bound_features.sh，新增 shell/webvid_ngfix_ef_grid.sh。
必须同时复制这三个文件到服务器 shell/ 下。
上轮 original 计时、搜索状态快照等 C++ 更新仍需先同步和编译。

```bash
set -euo pipefail
export ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
export DATA=/data/RoarGraph-main/data/clip-webvid-2.5M
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 清除以前的临时 ef 覆盖，恢复公开代码的 K-specific 网格。
unset EF_VALUES EF_VALUES_K10 EF_VALUES_K100 TRACE_EF TRACE_EFS

cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target bench_hnsw_unbounded_clean test_unbounded_clean_search search_hnsw_ngfix -j 4
"$ROOT/build/test/test_unbounded_clean_search"
```

性能实验：original 逐查询计时，1 轮；与机制诊断分开。

```bash
SUITE_OUT="$ROOT/result/graph_k_original_ngfixgrid_1k_v1" \
CPU=1 QUERY_LIMIT=1000 ROUNDS=1 TIMING_PROTOCOL=original RUN_DIAGNOSTICS=0 \
bash "$ROOT/shell/search_webvid_graph_k_matrix.sh"
```

性能结果自动汇总到 SUITE_OUT/comparison，共 144 个测试点：
两张图 × 两种搜索 × (20+16 个 ef)。参考验证仍保留。
如需增加重复次数，将 ROUNDS 改为 2，并使用新输出目录。

机制实验：每个图/K/ef 都重新跑同 ef 的 beam 和二跳，记录完整事件、
NDC 分解、恢复/丢失 GT、边类型、局部结构和 pre-probe 搜索状态。
分析依赖 pandas/numpy；先用 import 命令确认服务器已安装。

```bash
python3 -c 'import pandas,numpy'

OUT="$ROOT/result/bound_features_ngfixgrid_1k_v1" \
CPU=1 TRACE_LIMIT=1000 \
bash "$ROOT/shell/trace_webvid_bound_features.sh"
```

共 72 组配对诊断：两张图 × (20+16 个 ef)，每组 1000 条查询。
这些是重复使用同一批 1000 条，不是 72k 个不同查询。
机制结果按图、K、ef 分目录，例如：

bound_features_ngfixgrid_1k_v1/ngfix_k100/ef200/

该目录中 trace.csv 为配对主统计，trace_jump_events.csv、
trace_jump_mids.csv、trace_probe_state.csv 为明细，summary/、features/
为聚合统计。trace_plan.log 每开始一个 ef 就打印并记录进度。

两条主命令串行执行，避免测速与诊断争抢 CPU/内存。
增加到整组 ef 的 1k 详细事件后，机制文件与运行时间会明显增加，
不再是以前的四组 50 条小诊断。输出目录已存在时改 v1 为 v2。
本地环境无 Bash/Linux C++；网格内容已经核对，服务器仍需执行编译
和等价性检查。未删除旧方法、旧脚本或旧结果。
