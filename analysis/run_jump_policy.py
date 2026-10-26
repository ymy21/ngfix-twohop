#!/usr/bin/env python3
"""Collect original traces or rerun trained gates on disjoint queries (server only)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys


def vector_shape(path):
    with path.open("rb") as stream:
        n, d = struct.unpack("<II", stream.read(8))
    if not n or not d or path.stat().st_size < 8 + 4 * n * d:
        raise ValueError(f"Incomplete/invalid vector file: {path}")
    return n, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["collect", "evaluate"])
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=Path)
    ap.add_argument("--query-offset", type=int, default=0)
    ap.add_argument("--query-limit", type=int, default=1000)
    ap.add_argument("--efs", default="100,200")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--metric", choices=["ip_float", "l2_float"], default="ip_float")
    ap.add_argument("--cpu", default="1", help="Empty string disables taskset")
    ap.add_argument("--only", default="", help="Comma-separated learned case names; controls still run")
    ap.add_argument("--nn-cap", type=int, default=0, help="Apply an extra NDC cap to learned cases; adds cap-only control")
    args = ap.parse_args()
    efs = sorted(set(map(int, args.efs.split(","))))
    if not efs or min(efs) < args.k or args.k < 1 or args.query_offset < 0 or args.query_limit <= 0 or args.nn_cap < 0:
        raise ValueError("Need ef >= k > 0, offset >= 0, query-limit > 0, nn-cap >= 0")
    qn, qd = vector_shape(args.query)
    gn, gd = vector_shape(args.gt)
    with args.index.open("rb") as stream:
        index_header = list(struct.unpack("<7Q", stream.read(56)))
    if qn != gn or qd != index_header[5] or gd < args.k or args.query_offset >= qn:
        raise ValueError("Query/GT/index shapes or offset do not match")
    if not 0 < index_header[3] <= index_header[6] or args.index.stat().st_size < 56 + index_header[6] * (4 * qd + 1):
        raise ValueError("Index header/vector payload invalid or copy incomplete")
    query_count = min(args.query_limit, qn - args.query_offset)
    selected_ids = set(range(args.query_offset, args.query_offset + query_count))
    prefix = []
    if args.cpu:
        if not shutil.which("taskset"):
            raise ValueError("taskset unavailable; pass --cpu '' or set CPU=none")
        prefix = ["taskset", "-c", args.cpu]
    if not args.bin.is_file() or not os.access(args.bin, os.X_OK):
        raise ValueError(f"Build the search executable first: {args.bin}")
    help_text = subprocess.run([str(args.bin.resolve()), "--help"], check=True, text=True, capture_output=True).stdout
    if "--jump_policy" not in help_text or "--query_offset" not in help_text:
        raise ValueError("Binary lacks learned policy options; rebuild")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh output directory")
    common = prefix + [str(args.bin.resolve()), "--test_query_path", str(args.query.resolve()),
        "--test_gt_path", str(args.gt.resolve()), "--index_path", str(args.index.resolve()),
        "--metric", args.metric, "--K", str(args.k), "--query_offset", str(args.query_offset),
        "--query_limit", str(query_count), "--check_norms", "0", "--write_graph_audit", "0"]
    cases = []
    if args.action == "collect":
        if args.only or args.nn_cap:
            raise ValueError("Collection uses the original ungated policy; --only/--nn-cap are evaluation options")
        for ef in efs:
            cases.append((f"trace_ef{ef}", ["--search_mode", "boundtwohopcmp", "--event_trace_ef", str(ef),
                "--event_trace_query_limit", str(query_count), "--event_trace_compact", "1"]))
    else:
        if args.models is None:
            raise ValueError("evaluate requires --models")
        plan = json.loads((args.models / "plan.json").read_text(encoding="utf-8"))
        if plan["version"] != 1 or plan["k"] != args.k or plan["index_header"] != index_header or plan["index_bytes"] != args.index.stat().st_size:
            raise ValueError("Model K/index metadata do not match. Use the same unchanged index as training.")
        if plan.get("metric") is not None and plan["metric"] != args.metric:
            raise ValueError("Evaluation metric differs from policy training")
        if selected_ids.intersection(plan["trace_query_ids"]):
            raise ValueError("Evaluation overlaps training/validation traces. Choose disjoint QUERY_OFFSET/QUERY_LIMIT.")
        ef_args = ["--ef_values", ",".join(map(str, efs))]
        cases = [("baseline", ["--search_mode", "baseline"] + ef_args),
                 ("original", ["--search_mode", "boundtwohopcmp"] + ef_args),
                 ("gap0.08", ["--search_mode", "boundtwohopcmp", "--jump_mid_gap", "0.08"] + ef_args),
                 ("gap0.10", ["--search_mode", "boundtwohopcmp", "--jump_mid_gap", "0.10"] + ef_args),
                 ("cur100_gap0.08", ["--search_mode", "boundtwohopcmp", "--jump_cur_topk", "100", "--jump_mid_gap", "0.08"] + ef_args)]
        if args.nn_cap:
            cases.append((f"nncap{args.nn_cap}", ["--search_mode", "boundtwohopcmp",
                         "--jump_max_nn_per_query", str(args.nn_cap)] + ef_args))
        only = set(args.only.split(",")) if args.only else set()
        if only - {case["name"] for case in plan["cases"]}:
            raise ValueError(f"Unknown --only names: {only}")
        for case in plan["cases"]:
            if only and case["name"] not in only:
                continue
            options = list(case["args"])
            # Keep a copied model directory portable across server paths.
            for flag, name in [("--jump_tree_path", "tree.txt"), ("--jump_history_path", "history.txt")]:
                if flag in options:
                    options[options.index(flag) + 1] = str((args.models / name).resolve())
            name = case["name"]
            if args.nn_cap:
                options += ["--jump_max_nn_per_query", str(args.nn_cap)]
                name += f"_nncap{args.nn_cap}"
            cases.append((name, ["--search_mode", "boundtwohopcmp"] + ef_args + options))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(action=args.action, query=str(args.query.resolve()), gt=str(args.gt.resolve()),
                    query_offset=args.query_offset, query_count=query_count, k=args.k, metric=args.metric,
                    efs=efs, index=str(args.index.resolve()), index_header=index_header,
                    index_bytes=args.index.stat().st_size, models=str(args.models))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (args.out / "commands.txt").open("w", encoding="utf-8") as command_log:
        for name, options in cases:
            command = common + options + ["--result_path", str((args.out / f"{name}.csv").resolve())]
            command_log.write(shlex.join(command) + "\n")
            command_log.flush()
            print(f"Running {name}", flush=True)
            with (args.out / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    log.write(line)
                    print(line, end="", flush=True)
                if process.wait():
                    raise RuntimeError(f"{name} failed; see its log. Use a fresh OUT for a rerun.")
    if args.action == "evaluate":
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_jump_gates.py")),
                        "--indir", str(args.out)], check=True)
    print(f"Finished: {args.out}")


if __name__ == "__main__":
    main()
