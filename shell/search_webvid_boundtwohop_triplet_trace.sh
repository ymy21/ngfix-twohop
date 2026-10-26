#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
TRACE_EF=${TRACE_EF:-100}
TRACE_QUERY_LIMIT=${TRACE_QUERY_LIMIT:-1000}
KMEANS=${KMEANS:-0}
CPU=${CPU:-1}

QUERY=${QUERY:-${DATA}/query.10k.fbin}
GT=${GT:-${DATA}/gt.10k.ibin}
INDEX=${INDEX:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
BIN=${BIN:-${ROOT}/build/test/search_hnsw_ngfix}
OUT=${OUT:-${ROOT}/result/webvid_triplet_events/ef${TRACE_EF}_q${TRACE_QUERY_LIMIT}}
BASE=${OUT}/ngfix_boundtwohopcmp_events_k100

mkdir -p "${OUT}"

echo "========================================"
echo "WebVid boundtwohop triplet trace"
echo "ef=${TRACE_EF}, trace_queries=${TRACE_QUERY_LIMIT}, out=${OUT}"
echo "========================================"

taskset -c "${CPU}" "${BIN}" \
    --test_query_path "${QUERY}" \
    --test_gt_path "${GT}" \
    --metric ip_float \
    --K 100 \
    --index_path "${INDEX}" \
    --search_mode boundtwohopcmp \
    --event_trace_ef "${TRACE_EF}" \
    --event_trace_query_limit "${TRACE_QUERY_LIMIT}" \
    --result_path "${BASE}.csv" \
    2>&1 | tee "${BASE}.log"

python3 "${ROOT}/analysis/analyze_boundtwohop_events.py" \
    --result "${BASE}.csv" \
    --events "${BASE}_jump_events.csv" \
    --mids "${BASE}_jump_mids.csv" \
    --hopbins "${BASE}_jump_hopbins.csv" \
    --outdir "${OUT}/analysis_search"

python3 "${ROOT}/analysis/analyze_triplet_structure.py" \
    --events "${BASE}_jump_events.csv" \
    --mids "${BASE}_jump_mids.csv" \
    --outdir "${OUT}/analysis_triplet" \
    --kmeans "${KMEANS}"

echo "Done."
echo "Search-level analysis: ${OUT}/analysis_search"
echo "Triplet structural analysis: ${OUT}/analysis_triplet"
