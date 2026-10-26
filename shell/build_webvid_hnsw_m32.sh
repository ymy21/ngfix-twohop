#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/RoarGraph-main/ZHENGXIN/ngfix/NGFix}
DATA=${DATA:-/data/RoarGraph-main/data/clip-webvid-2.5M}
M32_INDEX=${M32_INDEX:-${ROOT}/data/_HNSWBottom_M32_efC2000_MEX48.index}
[[ ! -e ${M32_INDEX} ]] || { echo "Index exists; refusing overwrite: ${M32_INDEX}" >&2; exit 1; }
[[ ! -e ${M32_INDEX}.building ]] || { echo "Partial index exists: ${M32_INDEX}.building" >&2; exit 1; }
mkdir -p "$(dirname "${M32_INDEX}")"
# Existing builder uses 32 construction threads explicitly. OMP_NUM_THREADS
# cannot change its num_threads(32). No training queries or repair is applied.
"${ROOT}/build/test/build_hnsw_bottom" \
    --base_data_path "${BASE:-${DATA}/base.2.5M.fbin}" --metric ip_float \
    --result_hnsw_index_path "${M32_INDEX}.building" --M 32 --MEX 48 --efC 2000 \
    2>&1 | tee "${M32_INDEX}.build.log"
[[ -s ${M32_INDEX}.building ]] || { echo "Builder did not write an index" >&2; exit 1; }
mv -n -- "${M32_INDEX}.building" "${M32_INDEX}"
stat -c '%n %s bytes' "${M32_INDEX}"
