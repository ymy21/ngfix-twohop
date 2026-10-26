#!/usr/bin/env python3
"""Train tiny pre-probe policies. Fixed-trace screening is NOT rerun recall/NDC."""
import argparse
import glob
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor, export_text


FEATURE_NAMES = ["mid_gap_ratio", "cur_margin_ratio", "hop_over_ef", "first_base",
                 "first_ngfix", "first_rfix", "first_finite_eh", "first_inf"]
FEATURE_COLS = [f"f{i}" for i in range(8)]
LABEL_COLS = ["query_id", "ef", "cur_id", "mid_id", "fresh_nn_cmps", "useful_mid",
              "direct_new_gt", "nearest_recovered_desc_gt", "admitted_nn"]


def read_trace(path, k):
    header = pd.read_csv(path, nrows=0).columns
    exact = set(FEATURE_COLS).issubset(header)
    legacy = ["cur_dist", "mid_dist", "bound_at_reject", "hop", "first_edge_type", "first_edge_eh_raw"]
    cols = LABEL_COLS + (FEATURE_COLS if exact else legacy)
    frame = pd.read_csv(path, usecols=cols)
    if not exact:
        # Old CSVs round distances. Use only for a pilot; new compact traces
        # export the same float32 features the C++ model sees at inference.
        scale = np.maximum(frame.bound_at_reject.abs().to_numpy(), 1e-6)
        edge, eh = frame.first_edge_type, frame.first_edge_eh_raw
        features = np.column_stack([
            (frame.mid_dist - frame.bound_at_reject) / scale,
            (frame.bound_at_reject - frame.cur_dist) / scale,
            np.maximum(frame.hop - 1, 0) / np.maximum(frame.ef, k),
            edge.eq(0), edge.eq(1), edge.eq(2),
            np.where(edge.eq(1) & eh.ne(65535), eh, 0), edge.eq(1) & eh.eq(65535)])
        frame[FEATURE_COLS] = features.astype(np.float32)
        frame = frame[LABEL_COLS + FEATURE_COLS]
    frame[FEATURE_COLS] = frame[FEATURE_COLS].astype(np.float32)
    return frame, exact


def export_tree(model, path, k):
    tree = model.tree_
    lines = [f"NGFIX_JUMP_TREE_V1 8 {tree.node_count} {k}"]
    for i in range(tree.node_count):
        leaf = tree.children_left[i] < 0
        # Store scores as float32 to agree exactly with C++ inference.
        score = float(np.float32(np.clip(tree.value[i, 0, 0], 0.0, 1.0)))
        lines.append(f"{-1 if leaf else tree.feature[i]} {tree.children_left[i]} "
                     f"{tree.children_right[i]} {tree.threshold[i]:.17g} {score:.9g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def exported_predict(path, features):
    """Check the plain-text export's branching/float conversion before shipping."""
    rows = np.loadtxt(path, skiprows=1, ndmin=2)
    result = np.empty(len(features), dtype=np.float32)
    pending = [(0, np.arange(len(features)))]
    while pending:
        node, positions = pending.pop()
        f, left, right, threshold, score = rows[node]
        if f < 0:
            result[positions] = np.float32(score)
        else:
            mask = features[positions, int(f)].astype(np.float64) <= threshold
            pending.append((int(left), positions[mask]))
            pending.append((int(right), positions[~mask]))
    return result


def metrics(frame, keep):
    useful = frame.useful_mid.to_numpy()
    direct = frame.direct_new_gt.to_numpy()
    ndc = frame.fresh_nn_cmps.to_numpy()
    return dict(probe_keep=float(np.mean(keep)),
                original_nn_work_keep=float(ndc[keep].sum() / max(1, ndc.sum())),
                useful_probe_keep=float(useful[keep].sum() / max(1, useful.sum())),
                direct_new_event_keep=float(direct[keep].sum() / max(1, direct.sum())))


def threshold_candidates(scores):
    unique = np.unique(scores[scores >= 0])
    if len(unique) > 64:
        unique = np.unique(np.quantile(unique, np.linspace(0, 1, 64)))
    maximum = float(unique[-1]) if len(unique) else 0.0
    return np.unique(np.r_[0.0, unique, np.nextafter(maximum, np.inf)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mids", nargs="+", required=True)
    ap.add_argument("--index", type=Path, required=True, help="Exact index used to collect these traces")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--validation-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--history-min-queries", type=int, default=3)
    ap.add_argument("--history-prior-cost", type=float, default=1000.0)
    ap.add_argument("--retentions", default="0.90,0.95,0.98")
    args = ap.parse_args()
    if not (1 <= args.max_depth <= 6 and args.min_leaf >= 1 and args.k >= 1
            and 0 < args.validation_fraction < 1 and args.history_min_queries >= 1
            and np.isfinite(args.history_prior_cost) and args.history_prior_cost >= 0):
        raise ValueError("Invalid training options")
    retentions = sorted(set(map(float, args.retentions.split(","))))
    if not retentions or not all(0 < r <= 1 for r in retentions):
        raise ValueError("Retentions must be in (0,1]")
    paths = sorted({Path(p).resolve() for pattern in args.mids for p in glob.glob(pattern)})
    if not paths:
        raise ValueError("No trace files matched --mids")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise ValueError("Use a fresh model directory")
    with args.index.open("rb") as stream:
        index_header = list(struct.unpack("<7Q", stream.read(56)))
    node_count = index_header[3]
    if not 0 < node_count <= index_header[6]:
        raise ValueError("Invalid Linux 64-bit index header")
    manifests = []
    for parent in sorted({path.parent for path in paths}):
        manifest_path = parent / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("action") != "collect" or manifest["k"] != args.k
                    or manifest["index_header"] != index_header
                    or manifest["index_bytes"] != args.index.stat().st_size):
                raise ValueError("Trace collection manifest does not match K/index")
            manifests.append(manifest)
    if manifests and len({(m["query"], m["gt"], m["metric"]) for m in manifests}) != 1:
        raise ValueError("Do not combine different query files/metrics using overlapping row IDs")
    frames, exact_flags = zip(*(read_trace(path, args.k) for path in paths))
    data = pd.concat(frames, ignore_index=True)
    del frames
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Empty/nonfinite training trace")
    if data.duplicated(["query_id", "ef", "mid_id"]).any():
        raise ValueError("Duplicate query/ef/mid; do not mix different runs of the same trace")
    integers = data[LABEL_COLS].to_numpy()
    if not (integers == np.floor(integers)).all() or (integers < 0).any():
        raise ValueError("Trace IDs, labels and counts must be non-negative integers")
    if (data.cur_id >= node_count).any() or (data.mid_id >= node_count).any() or (data.ef < args.k).any():
        raise ValueError("Trace index mismatch or ef < k (unsupported for policy training)")
    expected_useful = (data.direct_new_gt > 0) | (data.nearest_recovered_desc_gt > 0)
    if not np.array_equal(data.useful_mid.to_numpy(), expected_useful.astype(int).to_numpy()):
        raise ValueError("useful_mid must mean direct-new OR nearest recovered descendant")
    if data.query_id.nunique() < 4:
        raise ValueError("Need at least four distinct training/validation queries")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(data, groups=data.query_id))
    train, validation = data.iloc[train_idx], data.iloc[val_idx]
    if not train.useful_mid.sum() or not validation.useful_mid.sum():
        raise ValueError("No useful probes in train/validation; collect more queries")
    x_train = train[FEATURE_COLS].to_numpy(dtype=np.float32)
    x_val = validation[FEATURE_COLS].to_numpy(dtype=np.float32)
    cost = np.maximum(1, train.fresh_nn_cmps.to_numpy(dtype=np.float64))
    utility = train.useful_mid.to_numpy(dtype=np.float64)
    # Weighted squared-loss regression: each leaf predicts sum(useful)/sum(cost).
    # This is an attribution-based cost score, NOT a calibrated probability.
    model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_leaf,
                                  random_state=args.seed)
    model.fit(x_train, utility / cost, sample_weight=cost)
    args.outdir.mkdir(parents=True, exist_ok=True)
    tree_path = args.outdir / "tree.txt"
    export_tree(model, tree_path, args.k)
    # Runtime compares stored float scores with a double CLI threshold.
    tree_scores = exported_predict(tree_path, x_val).astype(np.float64)
    if not np.array_equal(tree_scores, np.clip(model.predict(x_val), 0, 1).astype(np.float32)):
        raise ValueError("Exported tree prediction mismatch")
    (args.outdir / "tree_rules.txt").write_text(export_text(model, feature_names=FEATURE_NAMES,
                                                          decimals=8), encoding="utf-8")
    global_rate = float(utility.sum() / cost.sum())
    hist = train.assign(cost=cost).groupby("cur_id").agg(
        useful=("useful_mid", "sum"), cost=("cost", "sum"), probes=("mid_id", "size"),
        queries=("query_id", "nunique"))
    hist["score"] = ((hist.useful + args.history_prior_cost * global_rate) /
                     (hist.cost + args.history_prior_cost)).astype(np.float32)
    hist.to_csv(args.outdir / "history_audit.csv")
    with (args.outdir / "history.txt").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"NGFIX_JUMP_HISTORY_V1 {node_count} {args.k} {len(hist)}\n")
        for cur, row in hist.iterrows():
            stream.write(f"{int(cur)} {row.score:.9g} {int(row.queries)}\n")
    supported = hist[hist.queries >= args.history_min_queries]
    history_scores = validation.cur_id.map(supported.score).fillna(-1).to_numpy(dtype=np.float64)
    known = history_scores >= 0
    curves = []
    for policy, scores in [("tree", tree_scores), ("history", history_scores)]:
        for threshold in threshold_candidates(scores):
            keep = scores >= threshold
            if policy == "history":
                keep |= ~known  # unknown allow; matches runtime and prevents silent cold-start loss
            curves.append(dict(policy=policy, threshold=float(threshold), **metrics(validation, keep)))
    curves = pd.DataFrame(curves)
    curves.to_csv(args.outdir / "fixed_trace_validation.csv", index=False)
    selected = {}
    for retention in retentions:
        for policy in ("tree", "history"):
            eligible = curves[(curves.policy == policy) & (curves.useful_probe_keep >= retention)]
            best = eligible.sort_values(["original_nn_work_keep", "threshold"], ascending=[True, False]).iloc[0]
            selected[policy, retention] = float(best.threshold)
    # Plan consumed as structured JSON by the runner; never source model text as shell code.
    cases = []
    for retention in retentions:
        tag = format(retention, ".4g").replace(".", "p")
        tt, ht = selected["tree", retention], selected["history", retention]
        tree_args = ["--jump_tree_path", str(tree_path.resolve()), "--jump_tree_threshold", format(tt, ".17g")]
        history_args = ["--jump_history_path", str((args.outdir / "history.txt").resolve()),
                        "--jump_history_threshold", format(ht, ".17g"),
                        "--jump_history_min_queries", str(args.history_min_queries)]
        cases.extend([
            dict(name=f"tree_r{tag}", args=["--jump_policy", "tree"] + tree_args),
            dict(name=f"history_r{tag}", args=["--jump_policy", "history"] + history_args),
            dict(name=f"hybrid_r{tag}", args=["--jump_policy", "hybrid"] + tree_args + history_args)])
        keep = (tree_scores >= tt) | (known & (history_scores >= ht))
        selected["hybrid", retention] = metrics(validation, keep)
    traced_ids = set(map(int, data.query_id.unique()))
    for manifest in manifests:
        traced_ids.update(range(manifest["query_offset"], manifest["query_offset"] + manifest["query_count"]))
    plan = dict(version=1, k=args.k, index=str(args.index.resolve()),
                index_bytes=args.index.stat().st_size, index_header=index_header,
                metric=manifests[0]["metric"] if manifests else None,
                trace_query_ids=sorted(traced_ids), cases=cases)
    (args.outdir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"query_id": sorted(map(int, data.query_id.unique()))}).assign(
        split=lambda f: np.where(f.query_id.isin(train.query_id.unique()), "train", "validation")
    ).to_csv(args.outdir / "query_split.csv", index=False)
    metadata = dict(args={key: str(val) for key, val in vars(args).items()},
                    sklearn=sklearn.__version__, features=FEATURE_NAMES, traces=list(map(str, paths)),
                    exact_features=all(exact_flags), train_rows=len(train), validation_rows=len(validation),
                    train_queries=int(train.query_id.nunique()), validation_queries=int(validation.query_id.nunique()),
                    tree_depth=model.get_depth(), history_supported_nodes=len(supported),
                    validation_history_probe_coverage=float(known.mean()), global_useful_per_ndc=global_rate)
    (args.outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = ["# Jump policy training", "", f"Training/validation queries: {metadata['train_queries']}/{metadata['validation_queries']}",
              f"Tree depth: {model.get_depth()}; history supported nodes: {len(supported)}",
              f"Validation history coverage (probes): {known.mean():.4%}",
              "", "The table below screens a FIXED original trajectory. It is NOT online recall or NDC savings.",
              "Scores mean useful-probe attribution per cost; labels do not prove a causal recall contribution.",
              "History includes failures and shrinks to the training global rate. Unknown cur passes in history-only; hybrid uses tree.",
              "Do not mix traces from different query files, graph versions or K. Test queries must be absent from all collected traces.", ""]
    if not all(exact_flags):
        report += ["WARNING: legacy rounded-distance features were reconstructed; use new compact traces for final training.", ""]
    for retention in retentions:
        report.append(f"Requested useful-probe retention {retention:.2%}: tree threshold={selected['tree', retention]:.9g}, "
                      f"history threshold={selected['history', retention]:.9g}, hybrid screening={selected['hybrid', retention]}")
    report += ["", "Next: run the exported plan on disjoint queries and compare measured recall/NDC against ordinary baseline at higher ef.",
               "Current search timings retain attribution maps; they are diagnostic, not production QPS."]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
