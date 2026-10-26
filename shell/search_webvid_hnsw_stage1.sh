#!/bin/bash
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
BASE=${BASE:-${ROOT}/data/_HNSWBottom_M16_efC2000_MEX48.index}
REPAIRED=${REPAIRED:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
GRAPH=${GRAPH:-hnsw}
K=${K:-100}
PROFILE_ARGS=()
case ${GRAPH} in
    hnsw) INDEX=${INDEX:-${BASE}}; GRAPH_LABEL=unrepaired_bottom ;;
    ngfix) INDEX=${INDEX:-${REPAIRED}}; GRAPH_LABEL=repaired_reference; PROFILE_ARGS=(--allow-repaired) ;;
    *) echo "GRAPH must be hnsw or ngfix" >&2; exit 1 ;;
esac
OUT=${OUT:-${ROOT}/result/${GRAPH}_k${K}_unbounded_20261013_v1}
CPU=${CPU:-1}
QUERY_LIMIT=${QUERY_LIMIT:-1000}
source "${ROOT}/shell/webvid_ngfix_ef_grid.sh"
if [[ ${K} == 10 ]]; then
    TRACE_EFS=${TRACE_EFS:-20 100 200 800}
elif [[ ${K} == 100 ]]; then
    TRACE_EFS=${TRACE_EFS:-100 200 400 800}
else
    echo "K must be 10 or 100" >&2; exit 1
fi
EF_VALUES=$(webvid_select_ef_grid "${K}")
ROUNDS=${ROUNDS:-5}
TIMING_PROTOCOL=${TIMING_PROTOCOL:-batch}
RUN_DIAGNOSTICS=${RUN_DIAGNOSTICS:-1}
TRACE_LIMIT=${TRACE_LIMIT:-1000}
TRACE_SEARCH_STATE=${TRACE_SEARCH_STATE:-1}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -e ${OUT} ]]; then
    echo "Use a new OUT directory; existing results are preserved: ${OUT}" >&2
    exit 1
fi
mkdir -p "${OUT}"
python3 "${ROOT}/analysis/verify_base_index.py" --base "${BASE}" --repaired "${REPAIRED}" \
    --output "${OUT}/index_pair_verification.json"
"${ROOT}/build/test/test_unbounded_clean_search"
python3 "${ROOT}/analysis/run_hnsw_stage1.py" \
    --bin "${ROOT}/build/test/bench_hnsw_unbounded_clean" \
    --index "${INDEX}" --query "${DATA}/query.10k.fbin" --gt "${DATA}/gt.10k.ibin" \
    --out "${OUT}/clean" --cpu "${CPU}" --limit "${QUERY_LIMIT}" \
    --k "${K}" --efs "${EF_VALUES}" --rounds "${ROUNDS}" \
    --timing-protocol "${TIMING_PROTOCOL}" "${PROFILE_ARGS[@]}"

if [[ ${RUN_DIAGNOSTICS} == 1 ]]; then
    mkdir -p "${OUT}/diagnostics"
    taskset -c "${CPU}" "${ROOT}/build/test/search_hnsw_ngfix" \
        --test_query_path "${DATA}/query.10k.fbin" --test_gt_path "${DATA}/gt.10k.ibin" \
        --metric ip_float --K "${K}" --index_path "${INDEX}" \
        --query_limit "${QUERY_LIMIT}" --ef_values "${EF_VALUES}" \
        --search_mode boundtwohopcmp --result_path "${OUT}/diagnostics/base_paired.csv" \
        2>&1 | tee "${OUT}/diagnostics/base_paired.log"
    python3 "${ROOT}/analysis/summarize_hnsw_mechanism.py" \
        --paired "${OUT}/diagnostics/base_paired.csv" --clean-dir "${OUT}/clean" \
        --graph-label "${GRAPH_LABEL}" --out "${OUT}/diagnostics/summary"
    for EF in ${TRACE_EFS}; do
        TRACE_PREFIX=${OUT}/diagnostics/base_trace_ef${EF}
        taskset -c "${CPU}" "${ROOT}/build/test/search_hnsw_ngfix" \
            --test_query_path "${DATA}/query.10k.fbin" --test_gt_path "${DATA}/gt.10k.ibin" \
            --metric ip_float --K "${K}" --index_path "${INDEX}" \
            --query_limit "${TRACE_LIMIT}" --ef_values "${EF}" --search_mode boundtwohopcmp \
            --event_trace_ef "${EF}" --event_trace_query_limit "${TRACE_LIMIT}" \
            --trace_search_state "${TRACE_SEARCH_STATE}" \
            --result_path "${TRACE_PREFIX}.csv" \
            2>&1 | tee "${TRACE_PREFIX}.log"
        python3 "${ROOT}/analysis/summarize_hnsw_mechanism.py" \
            --paired "${TRACE_PREFIX}.csv" --events "${TRACE_PREFIX}_jump_events.csv" \
            --graph-label "${GRAPH_LABEL}" --out "${TRACE_PREFIX}_summary"
    done
fi
