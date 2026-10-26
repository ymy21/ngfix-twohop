#!/bin/bash
# Four sequential runs. Each graph/K run contains beam and unbounded two-hop.
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
SUITE_OUT=${SUITE_OUT:-${ROOT}/result/graph_k_matrix_20261013_v1}
if [[ -e ${SUITE_OUT} ]]; then
    echo "Use a new SUITE_OUT; existing results are preserved: ${SUITE_OUT}" >&2
    exit 1
fi
mkdir -p "${SUITE_OUT}"
for TOPK in 10 100; do
    for GRAPH_NAME in hnsw ngfix; do
        GRAPH=${GRAPH_NAME} K=${TOPK} OUT=${SUITE_OUT}/${GRAPH_NAME}_k${TOPK} \
            bash "${ROOT}/shell/search_webvid_hnsw_stage1.sh"
    done
done
python3 "${ROOT}/analysis/compare_graph_k_matrix.py" --indir "${SUITE_OUT}"
