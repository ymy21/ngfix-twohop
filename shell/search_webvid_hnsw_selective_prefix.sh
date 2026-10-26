#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
SUITE_OUT=${SUITE_OUT:-${ROOT}/result/hnsw_selective_prefix_1k_v1}
[[ ! -e ${SUITE_OUT} ]] || { echo "Use a new SUITE_OUT: ${SUITE_OUT}" >&2; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source "${ROOT}/shell/webvid_ngfix_ef_grid.sh"
if [[ ${DRY_RUN:-0} != 1 ]]; then
    "${ROOT}/build/test/test_selective_prefix_search"
fi
for TOPK in ${KS:-10 100}; do
    EFS=$(webvid_select_ef_grid "${TOPK}")
    if [[ ${TOPK} == 10 ]]; then UB_EFS=${UNBOUNDED_EFS_K10:-20,100,200}; else UB_EFS=${UNBOUNDED_EFS_K100:-100,200,400}; fi
    ARGS=()
    [[ ${DRY_RUN:-0} != 1 ]] || ARGS+=(--dry-run)
    [[ ${RUN_UNBOUNDED:-1} != 0 ]] || ARGS+=(--skip-unbounded)
    "${PYTHON:-python3}" "${ROOT}/analysis/run_hnsw_selective_prefix.py" \
        --bin "${ROOT}/build/test/bench_hnsw_selective_prefix" \
        --unbounded-bin "${ROOT}/build/test/bench_hnsw_unbounded_clean" \
        --index "${INDEX:-${ROOT}/data/_HNSWBottom_M16_efC2000_MEX48.index}" \
        --query "${QUERY:-${DATA}/query.10k.fbin}" --gt "${GT:-${DATA}/gt.10k.ibin}" \
        --out "${SUITE_OUT}/k${TOPK}" --k "${TOPK}" --efs "${EFS}" --unbounded-efs "${UB_EFS}" \
        --offset "${QUERY_OFFSET:-1000}" --limit "${QUERY_LIMIT:-1000}" --rounds "${ROUNDS:-2}" \
        --verify-limit "${VERIFY_LIMIT:-1000}" --cpu "${CPU:-1}" \
        --timing-protocol "${TIMING_PROTOCOL:-original}" \
        --gap "${MID_GAP:-0.0266}" --cur-margin "${CUR_MARGIN:-0.0127}" \
        --mid-cap "${MID_CAP:-2}" --nn-limit "${NN_LIMIT:-8}" --nn-cap "${NN_CAP:-1024}" "${ARGS[@]}"
done
