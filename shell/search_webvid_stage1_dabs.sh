#!/usr/bin/env bash
set -euo pipefail

# Override these with environment variables if your paths differ.
NGFIX_ROOT=${NGFIX_ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
WEBVID_DATA=${WEBVID_DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
WEBVID_INDEX=${WEBVID_INDEX:-${NGFIX_ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
STAGE1_QUERY_LIMIT=${STAGE1_QUERY_LIMIT:-200}
STAGE1_CPU=${STAGE1_CPU:-1}
STAGE1_TAG=${STAGE1_TAG:-g000_0020_dense}
STAGE1_RUN_BASELINE=${STAGE1_RUN_BASELINE:-1}
STAGE1_OVERWRITE=${STAGE1_OVERWRITE:-0}
STAGE1_DABS_GAMMAS=${STAGE1_DABS_GAMMAS:-0,0.002,0.004,0.006,0.008,0.010,0.0125,0.015,0.0175,0.020}

QUERY=${WEBVID_DATA}/query.10k.fbin
GT=${WEBVID_DATA}/gt.10k.ibin
BIN=${NGFIX_ROOT}/build/test/search_hnsw_ngfix
OUT=${NGFIX_ROOT}/result/webvid_stage1

mkdir -p "${OUT}"

if [[ ! -f "${NGFIX_ROOT}/build/CMakeCache.txt" ]]; then
    cmake -S "${NGFIX_ROOT}" -B "${NGFIX_ROOT}/build"
fi
cmake --build "${NGFIX_ROOT}/build" --target search_hnsw_ngfix -j

COMMON=(
    --test_query_path "${QUERY}"
    --test_gt_path "${GT}"
    --metric ip_float
    --K 100
    --index_path "${WEBVID_INDEX}"
    --query_limit "${STAGE1_QUERY_LIMIT}"
    --norm_sample_size 10000
    --norm_tolerance 0.001
    --write_graph_audit 0
)

BASELINE=${OUT}/baseline_q${STAGE1_QUERY_LIMIT}.csv
DABS=${OUT}/dabs_${STAGE1_TAG}_q${STAGE1_QUERY_LIMIT}.csv
SUMMARY_PREFIX=${OUT}/stage1_${STAGE1_TAG}_q${STAGE1_QUERY_LIMIT}

if [[ "${STAGE1_RUN_BASELINE}" == "1" ]]; then
    taskset -c "${STAGE1_CPU}" "${BIN}" \
        "${COMMON[@]}" \
        --search_mode baseline \
        --ef_values 100,150,200,300,400,420,500,600,800,1000 \
        --result_path "${BASELINE}" \
        2>&1 | tee "${OUT}/baseline_q${STAGE1_QUERY_LIMIT}.log"
elif [[ ! -s "${BASELINE}" ]]; then
    echo "Missing reusable baseline: ${BASELINE}" >&2
    exit 2
fi

if [[ "${STAGE1_OVERWRITE}" != "1" ]] && \
   compgen -G "${OUT}/dabs_${STAGE1_TAG}_q${STAGE1_QUERY_LIMIT}*" >/dev/null; then
    echo "DABS output for tag '${STAGE1_TAG}' already exists; choose a new STAGE1_TAG" >&2
    echo "or set STAGE1_OVERWRITE=1 to replace it." >&2
    exit 2
fi

taskset -c "${STAGE1_CPU}" "${BIN}" \
    "${COMMON[@]}" \
    --search_mode dabs \
    --dabs_gammas "${STAGE1_DABS_GAMMAS}" \
    --result_path "${DABS}" \
    2>&1 | tee "${OUT}/dabs_${STAGE1_TAG}_q${STAGE1_QUERY_LIMIT}.log"

python3 "${NGFIX_ROOT}/analysis/summarize_stage1_dabs.py" \
    --baseline "${BASELINE}" \
    --dabs "${DABS}" \
    --output-prefix "${SUMMARY_PREFIX}"

echo "Norm audit: ${OUT}/dabs_${STAGE1_TAG}_q${STAGE1_QUERY_LIMIT}_norm_audit.csv"
echo "Stage-1 report: ${SUMMARY_PREFIX}_report.md"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               