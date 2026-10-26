# #!/bin/bash
# set -e

# ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
# DATA=/data/RoarGraph-main/data/clip-webvid-2.5M

# QUERY=${DATA}/query.10k.fbin
# GT=${DATA}/gt.10k.ibin

# INDEX=${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index

# BIN=${ROOT}/build/test/search_hnsw_ngfix
# OUT=${ROOT}/result/webvid

# mkdir -p ${OUT}

# echo "========================================"
# echo "WebVid NGFix baseline"
# echo "========================================"

# taskset -c 1 ${BIN} \
#     --test_query_path ${QUERY} \
#     --test_gt_path ${GT} \
#     --metric ip_float \
#     --K 100 \
#     --index_path ${INDEX} \
#     --search_mode baseline \
#     --result_path ${OUT}/ngfix_baseline_k100.csv \
#     2>&1 | tee ${OUT}/ngfix_baseline_k100.log


# echo
# echo "========================================"
# echo "WebVid NGFix Two-Hop Jump"
# echo "========================================"

# taskset -c 1 ${BIN} \
#     --test_query_path ${QUERY} \
#     --test_gt_path ${GT} \
#     --metric ip_float \
#     --K 100 \
#     --index_path ${INDEX} \
#     --search_mode twohop \
#     --result_path ${OUT}/ngfix_twohop_k100.csv \
#     2>&1 | tee ${OUT}/ngfix_twohop_k100.log


#!/bin/bash
set -euo pipefail

ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
DATA=/data/RoarGraph-main/data/clip-webvid-2.5M

QUERY=${DATA}/query.10k.fbin
GT=${DATA}/gt.10k.ibin
INDEX=${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index
BIN=${ROOT}/build/test/search_hnsw_ngfix
OUT=${ROOT}/result/webvid

mkdir -p "${OUT}"

echo "========================================"
echo "WebVid NGFix baseline measurement only"
echo "10k test queries; ef = 100,200,300,500,800,1000"
echo "========================================"

taskset -c 1 "${BIN}" \
    --test_query_path "${QUERY}" \
    --test_gt_path "${GT}" \
    --metric ip_float \
    --K 100 \
    --index_path "${INDEX}" \
    --search_mode baseline \
    --result_path "${OUT}/ngfix_baseline_measurement_k100.csv" \
    2>&1 | tee "${OUT}/ngfix_baseline_measurement_k100.log"

echo
echo "Outputs:"
echo "  ${OUT}/ngfix_baseline_measurement_k100.csv"
echo "    - QUERY-level recall/NDC + examined/scored/admitted/expanded counters"
echo "    - query-level EH buckets for each stage"
echo "  ${OUT}/ngfix_baseline_measurement_k100_graph_eh_distribution.csv"
echo "    - exact EH distribution of all materialized NGFix edges in the index"
echo "  ${OUT}/ngfix_baseline_measurement_k100_search_eh_distribution.csv"
echo "    - exact EH distribution consumed by 10k queries, per ef and stage"
echo "  ${OUT}/ngfix_baseline_measurement_k100.log"
