#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}"
ARGS=(--bin "${BIN:-$ROOT/build/test/bench_hnsw_ngfix_clean}"
  --query "${QUERY:-$DATA/query.10k.fbin}" --gt "${GT:-$DATA/gt.10k.ibin}"
  --index "${INDEX:-$ROOT/data/_HNSWNgfix200M16_efC2000_MEX48.index}"
  --out "${OUT:?Set OUT to a fresh output directory}"
  --offset "${QUERY_OFFSET:-0}" --limit "${QUERY_LIMIT:-10000}" --k "${K:-100}"
  --beam-efs "${BEAM_EFS:-100,120,140,160,180,200,220,240,260,280,300,325,350,375,400,450,500,550,600,700,800,900,1000}"
  --cheap-efs "${CHEAP_EFS:-100,110,120,130,140,150,175,200,225,250,275,300,325,350,375,400,450,500,550,600}"
  --rounds "${ROUNDS:-3}" --warmup-passes "${WARMUP_PASSES:-1}"
  --verify-limit "${VERIFY_LIMIT:-200}"
  --min-batch-ms "${MIN_BATCH_MS:-300}" --seed "${SEED:-20261008}" --cpu "${CPU:-1}"
  --targets "${TARGETS:-0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.992,0.995}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then ARGS+=(--dry-run); fi
python3 "$ROOT/analysis/run_clean_qps.py" "${ARGS[@]}"
