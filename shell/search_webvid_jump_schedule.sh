#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
SUITE=${SUITE:-selection}
cmd=("${PYTHON:-python3}" "${ROOT}/analysis/run_jump_schedule.py" "${SUITE}"
    --bin "${BIN:-${ROOT}/build/test/search_hnsw_ngfix}"
    --query "${QUERY:-${DATA}/query.10k.fbin}" --gt "${GT:-${DATA}/gt.10k.ibin}"
    --index "${INDEX:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}"
    --out "${OUT:-${ROOT}/result/jump_${SUITE}}"
    --offset "${QUERY_OFFSET:-1000}" --limit "${QUERY_LIMIT:-200}" --k "${K:-100}"
    --efs "${EFS:-100,200}" --baseline-efs "${BASELINE_EFS:-100,150,200,300,400,500,600,800,1000}"
    --gap "${GAP:-0.06}" --mid-cap "${MID_CAP:-2}" --nn-cap "${NN_CAP:-4000}"
    --nn-per-mid "${NN_PER_MID:-0}" --nn-per-cur "${NN_PER_CUR:-0}"
    --selection "${SELECTION:-gap}" --windows "${WINDOWS:-4,8}" --cpu "${CPU:-1}"
    --gammas "${GAMMAS:-0,0.002,0.004,0.006,0.008,0.010,0.0125,0.015}")
[[ "${DRY_RUN:-0}" != 1 ]] || cmd+=(--dry-run)
[[ "${INCLUDE_DABS:-0}" != 1 ]] || cmd+=(--include-dabs)
[[ -z "${CONTROLS_FROM:-}" ]] || cmd+=(--controls-from "${CONTROLS_FROM}")
"${cmd[@]}"
