#!/bin/bash
set -euo pipefail

ROOT=/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix
DATA=/data/RoarGraph-main/data/clip-webvid-2.5M

QUERY=${DATA}/query.10k.fbin
GT=${DATA}/gt.10k.ibin
INDEX=${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index
BIN=${ROOT}/build/test/search_hnsw_ngfix
OUT=${ROOT}/result/webvid_events

# First characterization pass: trace ef=100 only.  Use 1000 queries to keep
# candidate-level CSV manageable; increase to 10000 after checking the signal.
TRACE_EF=${TRACE_EF:-100}
TRACE_QUERY_LIMIT=${TRACE_QUERY_LIMIT:-1000}

mkdir -p "${OUT}"

echo "========================================"
echo "WebVid boundtwohopcmp event trace: ef=${TRACE_EF}, query_limit=${TRACE_QUERY_LIMIT}"
echo "========================================"

taskset -c 1 "${BIN}" \
    --test_query_path "${QUERY}" \
    --test_gt_path "${GT}" \
    --metric ip_float \
    --K 100 \
    --index_path "${INDEX}" \
    --search_mode boundtwohopcmp \
    --event_trace_ef "${TRACE_EF}" \
    --event_trace_query_limit "${TRACE_QUERY_LIMIT}" \
    --result_path "${OUT}/ngfix_boundtwohopcmp_events_k100.csv" \
    2>&1 | tee "${OUT}/ngfix_boundtwohopcmp_events_k100.log"
