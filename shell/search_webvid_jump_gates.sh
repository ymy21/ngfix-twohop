#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Override paths through environment variables.
ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
QUERY=${QUERY:-${DATA}/query.10k.fbin}
GT=${GT:-${DATA}/gt.10k.ibin}
INDEX=${INDEX:-${ROOT}/data/_HNSWNgfix200M16_efC2000_MEX48.index}
BIN=${BIN:-${ROOT}/build/test/search_hnsw_ngfix}
OUT=${OUT:-${ROOT}/result/webvid_jump_gates}
QUERY_LIMIT=${QUERY_LIMIT:-1000}
QUERY_OFFSET=${QUERY_OFFSET:-0}
EFS=${EFS:-100,200,300,500}
K=${K:-100}
CPU=${CPU:-1}
SUITE=${SUITE:-core} # core | structure | budget | all
RECALL_DROP=${RECALL_DROP:-0.005} # absolute recall, 0.005 = 0.5 percentage points
PYTHON=${PYTHON:-python3}

case "${SUITE}" in core|structure|budget|all) ;; *) echo "Unknown SUITE: ${SUITE}" >&2; exit 2;; esac
for path in "${QUERY}" "${GT}" "${INDEX}"; do
    [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ -x "${BIN}" ]] || { echo "Build search_hnsw_ngfix first: ${BIN}" >&2; exit 2; }
"${BIN}" --help | grep -q -- '--jump_cur_topk' || { echo "Binary lacks jump gates; rebuild it first." >&2; exit 2; }
mkdir -p "${OUT}"
[[ ! -e "${OUT}/commands.txt" ]] || { echo "Use a fresh OUT directory: ${OUT}" >&2; exit 2; }

# Validate file headers and the minimum vector payload before the C++ loader.
# This index is the original Linux 64-bit, full-precision float index format.
"${PYTHON}" - "${QUERY}" "${GT}" "${INDEX}" "${K}" <<'PY'
import pathlib, struct, sys
query, gt, index = map(pathlib.Path, sys.argv[1:4])
def shape(path):
    with path.open('rb') as f:
        n, d = struct.unpack('<II', f.read(8))
    if not n or not d or path.stat().st_size < 8 + 4*n*d:
        raise SystemExit(f'Incomplete or invalid vector file: {path}')
    return n, d
qn, qd = shape(query)
gn, gd = shape(gt)
with index.open('rb') as f:
    m, m0, mex, n, ep, dim, capacity = struct.unpack('<7Q', f.read(56))
if qn != gn or qd != dim or not 0 < int(sys.argv[4]) <= gd:
    raise SystemExit('Query/GT/index dimensions or K do not match')
if not 0 < n <= capacity or ep >= n or index.stat().st_size < 56 + capacity*(dim*4+1):
    raise SystemExit('Index header/payload invalid or copy incomplete')
PY

run_case() {
    local name=$1 mode=$2
    shift 2
    local cmd=("${BIN}" --test_query_path "${QUERY}" --test_gt_path "${GT}"
        --index_path "${INDEX}" --metric ip_float --K "${K}"
        --query_limit "${QUERY_LIMIT}" --query_offset "${QUERY_OFFSET}" --ef_values "${EFS}"
        --check_norms 0 --write_graph_audit 0
        --search_mode "${mode}" --result_path "${OUT}/${name}.csv" "$@")
    if [[ -n "${CPU}" ]] && command -v taskset >/dev/null 2>&1; then
        cmd=(taskset -c "${CPU}" "${cmd[@]}")
    fi
    printf '%q ' "${cmd[@]}" >> "${OUT}/commands.txt"
    printf '\n' >> "${OUT}/commands.txt"
    echo "Running ${name}: ${mode} $*"
    "${cmd[@]}" 2>&1 | tee "${OUT}/${name}.log"
}

run_case baseline baseline
run_case original boundtwohopcmp

if [[ "${SUITE}" == structure || "${SUITE}" == budget ]]; then
    run_case gap0.08 boundtwohopcmp --jump_mid_gap 0.08
fi

if [[ "${SUITE}" == core || "${SUITE}" == all ]]; then
    for gap in 0.06 0.08 0.10; do
        run_case "gap${gap}" boundtwohopcmp --jump_mid_gap "${gap}"
    done
    for rank in 25 50 100; do
        run_case "cur${rank}" boundtwohopcmp --jump_cur_topk "${rank}"
        run_case "cur${rank}_gap0.08" boundtwohopcmp --jump_cur_topk "${rank}" --jump_mid_gap 0.08
    done
    run_case hop5 boundtwohopcmp --jump_min_hops 5
    run_case hop5_gap0.08 boundtwohopcmp --jump_min_hops 5 --jump_mid_gap 0.08
fi

if [[ "${SUITE}" == structure || "${SUITE}" == all ]]; then
    # Single-factor controls, then interactions with the same distance gate.
    for edge in base ngfix ngfix_finite ngfix_inf; do
        run_case "first_${edge}" boundtwohopcmp --jump_first_edge "${edge}"
        run_case "gap0.08_first_${edge}" boundtwohopcmp --jump_mid_gap 0.08 --jump_first_edge "${edge}"
    done
    run_case second_ngfix boundtwohopcmp --jump_second_edge ngfix
    run_case first_second_ngfix boundtwohopcmp --jump_first_edge ngfix --jump_second_edge ngfix
fi

if [[ "${SUITE}" == budget || "${SUITE}" == all ]]; then
    for cap in 2000 4000 8000; do
        run_case "nncap${cap}" boundtwohopcmp --jump_max_nn_per_query "${cap}"
        run_case "gap0.08_nncap${cap}" boundtwohopcmp --jump_mid_gap 0.08 --jump_max_nn_per_query "${cap}"
    done
    for cap in 2 4; do
        run_case "midcap${cap}" boundtwohopcmp --jump_max_mids_per_cur "${cap}"
    done
fi

"${PYTHON}" "${ROOT}/analysis/summarize_jump_gates.py" --indir "${OUT}" --recall-drop "${RECALL_DROP}"
echo "Results: ${OUT}/analysis"
