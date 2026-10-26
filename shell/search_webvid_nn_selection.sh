#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}"
ARGS=(--suite "${SUITE:-diagnostic}" --bin "${BIN:-$ROOT/build/test/search_hnsw_ngfix}"
  --query "${QUERY:-$DATA/query.10k.fbin}" --gt "${GT:-$DATA/gt.10k.ibin}"
  --index "${INDEX:-$ROOT/data/_HNSWNgfix200M16_efC2000_MEX48.index}"
  --out "${OUT:?Set OUT to a new output directory}"
  --offset "${QUERY_OFFSET:-1000}" --limit "${QUERY_LIMIT:-20}"
  --k "${K:-100}" --efs "${EFS:-100,200}"
  --baseline-efs "${BASELINE_EFS:-100,110,120,130,140,150,200,250,300,400,500,600,800,1000}"
  --gap "${GAP:-0.06}" --mid-cap "${MID_CAP:-2}" --nn-cap "${NN_CAP:-4000}"
  --nn-limit "${NN_LIMIT:-8}" --cheap-dims "${CHEAP_DIMS:-16}" --seeds "${SEEDS:-17}"
  --cpu "${CPU:-1}" --gammas "${GAMMAS:-0,0.0005,0.001,0.0015,0.002,0.004,0.006,0.008,0.010,0.0125,0.015}")
if [[ "${INCLUDE_DABS:-0}" == 1 ]]; then ARGS+=(--include-dabs); fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then ARGS+=(--dry-run); fi
python3 "$ROOT/analysis/run_jump_nn_selection.py" "${ARGS[@]}"
