"""Prepare disjoint, nested query-ID samples; this does not build A/B labels.

Counts are declarations unless --train-file/--base-file validate fbin headers.
Input files and graph data are never modified. No third-party dependencies.
"""
import argparse
import array
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
from decimal import Decimal, ROUND_HALF_UP


def audit_fbin(path, expected_count):
    if path is None:
        return {"verified": False, "declared_count": expected_count}
    source = Path(path).resolve()
    with source.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"Incomplete fbin header: {source}")
    count, dim = struct.unpack("<II", header)
    if count != expected_count or dim != 512:
        raise ValueError(f"Unexpected fbin shape {count} x {dim}: {source}")
    expected_bytes = 8 + count * dim * 4
    if source.stat().st_size != expected_bytes:
        raise ValueError(f"Unexpected fbin file length: {source}")
    return {"verified": True, "path": str(source), "count": count,
            "dimension": dim, "bytes": expected_bytes,
            "note": "Header and length verified; content and GT not audited."}


def write_ids(path, ids):
    if ids.itemsize != 4:
        raise RuntimeError("This platform does not provide 32-bit array('I').")
    if sys.byteorder != "little":
        ids.byteswap()
    with path.open("xb") as stream:
        ids.tofile(stream)
    return {"file": path.name, "count": len(ids),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--base-count", type=int, required=True)
    parser.add_argument("--train-file")
    parser.add_argument("--base-file")
    parser.add_argument("--dev-count", type=int, default=10000)
    parser.add_argument("--calibration-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--percentages", default="0.1,0.5,1,2,5,8,10,20,30")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.train_count < 2**32 or args.base_count <= 0:
        parser.error("Counts must be positive, with train count below 2**32.")
    if args.dev_count <= 0 or args.calibration_count <= 0:
        parser.error("Development and calibration sets must be nonempty.")
    percentages = sorted(set(Decimal(p) for p in args.percentages.split(",")))
    if any(not p.is_finite() or p <= 0 or p > 100 for p in percentages):
        parser.error("Percentages must be finite and in (0,100].")
    checkpoints = []
    for percentage in percentages:
        count = int((Decimal(args.base_count) * percentage / 100)
                    .quantize(Decimal(1), rounding=ROUND_HALF_UP))
        if count < 1:
            parser.error("A percentage rounded to zero queries.")
        checkpoints.append({"percent_of_base": str(percentage), "query_count": count})
    max_count = max(point["query_count"] for point in checkpoints)
    holdout_count = args.dev_count + args.calibration_count
    if max_count + holdout_count > args.train_count:
        parser.error("The requested largest checkpoint does not fit after holdouts.")
    audits = {"train": audit_fbin(args.train_file, args.train_count),
              "base": audit_fbin(args.base_file, args.base_count)}
    args.out.mkdir(parents=True, exist_ok=True)
    names = ("development_ids.u32", "calibration_ids.u32", "training_order.u32",
             "manifest.json")
    if any((args.out / name).exists() for name in names):
        parser.error("Output files already exist; choose a fresh directory.")
    # Shuffle the entire declared training row space. The holdouts and prefix
    # order remain fixed when the requested checkpoint list changes.
    permutation = array.array("I", range(args.train_count))
    random.Random(args.seed).shuffle(permutation)
    files = {
        "development": write_ids(args.out / names[0], permutation[:args.dev_count]),
        "calibration": write_ids(args.out / names[1],
                                 permutation[args.dev_count:holdout_count]),
        "training_order": write_ids(args.out / names[2],
                                    permutation[holdout_count:holdout_count + max_count]),
    }
    for point in checkpoints:
        point.update({"ids_file": names[2], "start_row": 0,
                      "read_first_ids": point["query_count"]})
    manifest = {
        "schema_version": 1, "purpose": "A/B FULL-NN training-size sweep input IDs",
        "seed": args.seed, "python_version": sys.version.split()[0],
        "id_format": "headerless little-endian uint32, zero-based source row IDs",
        "ratio_denominator": "base vector count, not remaining training pool",
        "base_count": args.base_count, "train_count": args.train_count,
        "source_audits": audits, "source_grouping_audited": False,
        "split_method": "row-random full permutation; all splits come from train workload",
        "training_pool_count": args.train_count - holdout_count,
        "files": files, "checkpoints": checkpoints,
        "experiment": {"graph": "fixed unrepaired HNSW M16 bottom",
                       "top_L": 200, "k": 100, "schedule": "after_onehop",
                       "mid_methods": ["A_union", "B_union"], "nn_selection": "FULL",
                       "primary_percentages": ["0.1", "0.5", "1", "2", "5"],
                       "extension_percentages": ["8", "10", "20", "30"]},
        "not_completed": ["train/base content and GT matching audit",
                          "source/near-duplicate grouping audit",
                          "A/B label construction", "online search integration",
                          "WebVid performance experiments"],
    }
    with (args.out / names[3]).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"out": str(args.out.resolve()), "seed": args.seed,
                      "checkpoints": checkpoints,
                      "source_files_verified": all(a["verified"] for a in audits.values())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
