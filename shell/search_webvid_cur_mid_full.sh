#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
SUITE_OUT=${SUITE_OUT:-${ROOT}/result/cur_mid_full_1k_v1}
[[ ! -e ${SUITE_OUT} ]] || { echo "Choose a new SUITE_OUT: ${SUITE_OUT}" >&2; exit 1; }
source "${ROOT}/shell/webvid_ngfix_ef_grid.sh"
if [[ ${DRY_RUN:-0} != 1 ]]; then
    "${ROOT}/build/test/test_cur_mid_full_search"
fi
for TOPK in ${KS:-10 100}; do
    EFS=$(webvid_select_ef_grid "${TOPK}")
    ARGS=()
    [[ ${DRY_RUN:-0} != 1 ]] || ARGS+=(--dry-run)
    if [[ ${RUN_M32:-1} == 1 ]]; then ARGS+=(--m32-index "${M32_INDEX:-${ROOT}/data/_HNSWBottom_M32_efC2000_MEX48.index}"); fi
    if [[ ${RUN_NGFIX:-1} == 1 ]]; then ARGS+=(--ngfix-index "${NGFIX_INDEX:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}"); fi
    "${PYTHON:-python3}" "${ROOT}/analysis/run_hnsw_cur_mid_full.py" \
        --bin "${ROOT}/build/test/bench_hnsw_cur_mid_full" \
        --index "${INDEX:-${ROOT}/data/_HNSWBottom_M16_efC2000_MEX48.index}" \
        --query "${QUERY:-${DATA}/query.10k.fbin}" --gt "${GT:-${DATA}/gt.10k.ibin}" \
        --out "${SUITE_OUT}/k${TOPK}" --k "${TOPK}" --efs "${EFS}" \
        --offset "${QUERY_OFFSET:-1000}" --limit "${QUERY_LIMIT:-1000}" --rounds "${ROUNDS:-2}" \
        --verify-limit "${VERIFY_LIMIT:-1000}" --cpu "${CPU:-1}" \
        --gap "${MID_GAP:-0.0266}" --cur-slack "${CUR_SLACK:-0.0127}" --mid-cap "${MID_CAP:-0}" \
        --profiles "${PROFILES:-mid_gap,cur_mid,ratio_1}" "${ARGS[@]}"
done
