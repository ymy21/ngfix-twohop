#!/bin/bash
# Run from any directory; verify the pair before starting experiments.
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
BASE=${BASE:-${ROOT}/data/_HNSWBottom_M16_efC2000_MEX48.index}
REPAIRED=${REPAIRED:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
BIN=${BIN:-${ROOT}/build/test/search_hnsw_ngfix}
OUT=${OUT:-${ROOT}/result/webvid_verified_base}
CPU=${CPU:-1}
EF_VALUES=${EF_VALUES:-100,150,200,300,400,600,800,1200,1600,2400}
NN_BUDGETS=${NN_BUDGETS:-64 128 256}
mkdir -p "${OUT}"

python3 "${ROOT}/analysis/verify_base_index.py" \
    --base "${BASE}" --repaired "${REPAIRED}" \
    --output "${OUT}/index_pair_verification.json" \
    | tee "${OUT}/index_pair_verification.log"

# Performance pass: independent test queries, no event tracing.
for GRAPH in base ngfix; do
    INDEX=${BASE}
    if [[ ${GRAPH} == ngfix ]]; then INDEX=${REPAIRED}; fi
    taskset -c "${CPU}" "${BIN}" \
        --test_query_path "${DATA}/query.10k.fbin" \
        --test_gt_path "${DATA}/gt.10k.ibin" \
        --metric ip_float --K 100 --index_path "${INDEX}" \
        --ef_values "${EF_VALUES}" --search_mode baseline \
        --result_path "${OUT}/${GRAPH}_baseline_k100.csv" \
        2>&1 | tee "${OUT}/${GRAPH}_baseline_k100.log"
    for BUDGET in ${NN_BUDGETS}; do
        taskset -c "${CPU}" "${BIN}" \
            --test_query_path "${DATA}/query.10k.fbin" \
            --test_gt_path "${DATA}/gt.10k.ibin" \
            --metric ip_float --K 100 --index_path "${INDEX}" \
            --ef_values "${EF_VALUES}" --search_mode boundtwohop \
            --jump_max_nn_per_query "${BUDGET}" \
            --result_path "${OUT}/${GRAPH}_boundtwohop_budget${BUDGET}_k100.csv" \
            2>&1 | tee "${OUT}/${GRAPH}_boundtwohop_budget${BUDGET}_k100.log"
    done
done

# Mechanism pass on the independently verified unrepaired graph only.
# Trace timings include instrumentation overhead; do not use them for QPS claims.
TRACE_EF=${TRACE_EF:-200}
TRACE_QUERY_LIMIT=${TRACE_QUERY_LIMIT:-1000}
TRACE_NN_BUDGET=${TRACE_NN_BUDGET:-128}
taskset -c "${CPU}" "${BIN}" \
    --test_query_path "${DATA}/query.10k.fbin" \
    --test_gt_path "${DATA}/gt.10k.ibin" \
    --metric ip_float --K 100 --index_path "${BASE}" \
    --ef_values "${TRACE_EF}" --search_mode boundtwohopcmp \
    --query_limit "${TRACE_QUERY_LIMIT}" \
    --event_trace_ef "${TRACE_EF}" \
    --event_trace_query_limit "${TRACE_QUERY_LIMIT}" \
    --jump_max_nn_per_query "${TRACE_NN_BUDGET}" \
    --result_path "${OUT}/base_mechanism_ef${TRACE_EF}_budget${TRACE_NN_BUDGET}_k100.csv" \
    2>&1 | tee "${OUT}/base_mechanism_ef${TRACE_EF}_budget${TRACE_NN_BUDGET}_k100.log"
