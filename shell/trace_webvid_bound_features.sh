#!/bin/bash
# Independent diagnostics only: does not rerun the clean QPS sweep.
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
OUT=${OUT:-${ROOT}/result/bound_features_20261013_v1}
CPU=${CPU:-1}
TRACE_LIMIT=${TRACE_LIMIT:-1000}
source "${ROOT}/shell/webvid_ngfix_ef_grid.sh"
BASE=${BASE:-${ROOT}/data/_HNSWBottom_M16_efC2000_MEX48.index}
REPAIRED=${REPAIRED:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -e ${OUT} ]]; then echo "Use a fresh OUT directory: ${OUT}" >&2;exit 1;fi
python3 -c "import pandas, numpy"
mkdir -p "${OUT}"
python3 "${ROOT}/analysis/verify_base_index.py" --base "${BASE}" --repaired "${REPAIRED}" \
    --output "${OUT}/index_pair_verification.json"
"${ROOT}/build/test/test_unbounded_clean_search"
for TOPK in 10 100; do
    # Legacy TRACE_EF (one value) and TRACE_EFS (spaces/commas) remain supported.
    # With neither set, diagnostics use the same grid as the performance run.
    TRACE_VALUES=${TRACE_EFS:-${TRACE_EF:-$(webvid_select_ef_grid "${TOPK}")}}
    TRACE_VALUES=${TRACE_VALUES//,/ }
    for GRAPH in hnsw ngfix; do
        INDEX=${BASE};LABEL=unrepaired_bottom
        if [[ ${GRAPH} == ngfix ]]; then INDEX=${REPAIRED};LABEL=repaired_reference;fi
        FOLDER=${OUT}/${GRAPH}_k${TOPK}
        mkdir -p "${FOLDER}"
        for EF in ${TRACE_VALUES}; do
            if [[ ! ${EF} =~ ^[0-9]+$ ]] || (( EF < TOPK )); then
                echo "Require integer ef>=K; got ef=${EF}, K=${TOPK}" >&2;exit 1
            fi
            EF_DIR=${FOLDER}/ef${EF}
            if [[ -e ${EF_DIR} ]]; then echo "Duplicate ef: ${EF}" >&2;exit 1;fi
            mkdir -p "${EF_DIR}"
            PREFIX=${EF_DIR}/trace
            printf 'graph=%s K=%s queries=%s ef=%s\n' "${GRAPH}" "${TOPK}" "${TRACE_LIMIT}" "${EF}" \
                | tee -a "${OUT}/trace_plan.log"
            taskset -c "${CPU}" "${ROOT}/build/test/search_hnsw_ngfix" \
                --test_query_path "${DATA}/query.10k.fbin" --test_gt_path "${DATA}/gt.10k.ibin" \
                --metric ip_float --K "${TOPK}" --index_path "${INDEX}" \
                --query_limit "${TRACE_LIMIT}" --ef_values "${EF}" --search_mode boundtwohopcmp \
                --event_trace_ef "${EF}" --event_trace_query_limit "${TRACE_LIMIT}" \
                --trace_search_state 1 --result_path "${PREFIX}.csv" \
                2>&1 | tee "${PREFIX}.log"
            python3 "${ROOT}/analysis/summarize_hnsw_mechanism.py" --paired "${PREFIX}.csv" \
                --events "${PREFIX}_jump_events.csv" --graph-label "${LABEL}" --out "${EF_DIR}/summary"
            python3 "${ROOT}/analysis/analyze_bound_features.py" --mids "${PREFIX}_jump_mids.csv" \
                --events "${PREFIX}_jump_events.csv" --states "${PREFIX}_probe_state.csv" --outdir "${EF_DIR}/features"
        done
    done
done
