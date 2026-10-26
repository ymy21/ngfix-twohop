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

for MODE in baseline boundtwohop boundtwohopcmp; do
  echo "========================================"
  echo "WebVid mode=${MODE}"
  echo "========================================"
  taskset -c 1 "${BIN}" \
      --test_query_path "${QUERY}" \
      --test_gt_path "${GT}" \
      --metric ip_float \
      --K 100 \
      --index_path "${INDEX}" \
      --search_mode "${MODE}" \
      --result_path "${OUT}/ngfix_${MODE}_k100.csv" \
      2>&1 | tee "${OUT}/ngfix_${MODE}_k100.log"
done
