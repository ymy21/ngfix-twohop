#!/usr/bin/env python3
"""Summarize actual reruns; compare paired queries and measured recall/NDC curves."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path):
    frame = pd.read_csv(path)
    if "row_type" not in frame:
        return None
    frame = frame[frame.row_type.eq("QUERY")].copy()
    if frame.empty:
        raise ValueError(f"No query rows: {path}")
    if frame.duplicated(["ef", "query_id"]).any():
        raise ValueError(f"Duplicate query keys: {path}")
    if not np.allclose(frame.ndc, frame.ndc_entry + frame.ndc_mid + frame.ndc_nn):
        raise ValueError(f"NDC accounting mismatch: {path}")
    if not np.allclose(frame.ndc_nn, frame.ndc_nn_jump + frame.ndc_nn_nojump):
        raise ValueError(f"NN NDC accounting mismatch: {path}")
    if "gate_rejected_mids" in frame:
        blocked = frame[["gate_blocked_hop", "gate_blocked_cur", "gate_blocked_gap",
                         "gate_blocked_edge", "gate_blocked_budget"]].sum(axis=1)
        if "gate_blocked_policy" in frame:
            blocked += frame.gate_blocked_policy
        if "gate_blocked_detour" in frame:
            blocked += frame.gate_blocked_detour
        for c in ["gate_blocked_selection", "gate_blocked_trigger", "gate_blocked_batch_budget"]:
            if c in frame: blocked += frame[c]
        if not np.allclose(frame.gate_rejected_mids, blocked + frame.twohop_mid_scans):
            raise ValueError(f"Gate accounting mismatch: {path}")
    return frame.set_index(["ef", "query_id"]).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, help="Default: INDIR/analysis")
    ap.add_argument("--recall-drop", type=float, default=0.005)
    ap.add_argument("--targets", default="0.95,0.98,0.99,0.995")
    args = ap.parse_args()
    if not np.isfinite(args.recall_drop) or not 0 <= args.recall_drop <= 1:
        raise ValueError("recall-drop must be within [0,1]")
    runs = {}
    for path in sorted(args.indir.glob("*.csv")):
        # Exclude the instrumentation sidecars written by the search executable.
        if any(path.stem.endswith(s) for s in ["_search_eh_distribution", "_norm_audit",
               "_graph_eh_distribution", "_repair_degree_distribution", "_repair_degree_summary",
               "_repair_eh_by_saturation", "_jump_events", "_jump_mids", "_jump_hopbins", "_nn_candidates"]):
            continue
        frame = load_run(path)
        if frame is not None:
            runs[path.stem] = frame
    if "original" not in runs or "baseline" not in runs:
        raise ValueError("Need baseline.csv and original.csv from the same sweep")
    reference = runs["original"]
    query_ids = reference.index.get_level_values("query_id").unique().sort_values()
    for name, frame in runs.items():
        for ef, group in frame.groupby(level="ef"):
            if not group.index.get_level_values("query_id").sort_values().equals(query_ids):
                raise ValueError(f"Different query IDs at ef={ef}: {name}")
    if not reference.index.isin(runs["baseline"].index).all():
        raise ValueError("Baseline must include every original query/ef pair")
    metrics = ["recall", "ndc", "ndc_mid", "ndc_nn", "ndc_nn_nojump", "twohop_mid_scans",
               "jump_mids", "jump_candidates", "jump_candidates_gt", "recovered_gt_total",
               "recovered_gt_direct_jump", "recovered_gt_non_gt_bridge_root", "lost_gt_total", "latency_us"]
    metrics += sorted({c for frame in runs.values() for c in frame.columns if c.startswith("gate_")})
    summaries, paired = [], []
    for name, frame in runs.items():
        if name != "baseline" and not frame.index.equals(reference.index):
            raise ValueError(f"Query/ef set differs from original: {name}")
        if name != "baseline":
            baseline = runs["baseline"].loc[frame.index]
            if not np.allclose(frame.paired_baseline_recall, baseline.recall) or not np.array_equal(
                    frame.paired_baseline_ndc.to_numpy(), baseline.ndc.to_numpy()):
                raise ValueError(f"Paired baseline differs (inputs/mode changed?): {name}")
        summary = frame.groupby(level="ef")[[c for c in metrics if c in frame]].mean().reset_index()
        summary.insert(0, "config", name)
        summary["queries"] = frame.groupby(level="ef").size().to_numpy()
        summary["jump_gt_fraction"] = summary.jump_candidates_gt / summary.jump_candidates.replace(0, np.nan)
        summaries.append(summary)
        for ef, group in frame.groupby(level="ef"):
            if not group.index.isin(reference.index).all():
                continue  # Extra baseline ef points belong in the recall curve, not paired deltas.
            ref = reference.loc[group.index]
            dr = group.recall - ref.recall
            dn = group.ndc - ref.ndc
            paired.append(dict(config=name, ef=ef, queries=len(group), recall=group.recall.mean(),
                ndc=group.ndc.mean(), recall_delta=dr.mean(),
                recall_delta_se=dr.std(ddof=1) / np.sqrt(len(group)), ndc_delta=dn.mean(),
                ndc_delta_se=dn.std(ddof=1) / np.sqrt(len(group)),
                ndc_reduction=1 - group.ndc.mean() / ref.ndc.mean(),
                probe_reduction=1 - group.twohop_mid_scans.mean() / ref.twohop_mid_scans.mean()
                    if ref.twohop_mid_scans.mean() else np.nan,
                jump_reduction=1 - group.jump_candidates.mean() / ref.jump_candidates.mean()
                    if ref.jump_candidates.mean() else np.nan,
                recall_tolerance_pass=dr.mean() >= -args.recall_drop))
    out = args.outdir or args.indir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    points = pd.concat(summaries, ignore_index=True)
    points.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "same_ef_vs_original.csv", index=False)
    # Discrete measured points only: no recall interpolation/extrapolation.
    targets = []
    for target in map(float, args.targets.split(",")):
        if not np.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("Recall targets must be within [0,1]")
        for name, group in points.groupby("config"):
            qualified = group[group.recall >= target]
            row = dict(target_recall=target, config=name, reached=not qualified.empty,
                       ef=np.nan, recall=np.nan, ndc=np.nan)
            if not qualified.empty:
                best = qualified.loc[qualified.ndc.idxmin()]
                row.update(ef=best.ef, recall=best.recall, ndc=best.ndc)
            targets.append(row)
    pd.DataFrame(targets).to_csv(out / "at_target_recall.csv", index=False)
    pareto = []
    for _, point in points.iterrows():
        dominated = ((points.recall >= point.recall) & (points.ndc <= point.ndc) &
                     ((points.recall > point.recall) | (points.ndc < point.ndc))).any()
        pareto.append(not dominated)
    points[pareto].sort_values("recall").to_csv(out / "pareto.csv", index=False)
    baseline_points = points[points.config.eq("baseline")]
    dominance = []
    for _, point in points[~points.config.eq("baseline")].iterrows():
        better = baseline_points[(baseline_points.recall >= point.recall) &
                                 (baseline_points.ndc <= point.ndc) &
                                 ((baseline_points.recall > point.recall) |
                                  (baseline_points.ndc < point.ndc))]
        row = dict(config=point.config, ef=point.ef, recall=point.recall, ndc=point.ndc,
                   dominated_by_baseline=not better.empty, baseline_ef=np.nan,
                   baseline_recall=np.nan, baseline_ndc=np.nan, ndc_ratio=np.nan)
        if not better.empty:
            best = better.loc[better.ndc.idxmin()]
            row.update(baseline_ef=best.ef, baseline_recall=best.recall,
                       baseline_ndc=best.ndc, ndc_ratio=point.ndc / best.ndc)
        dominance.append(row)
    pd.DataFrame(dominance).to_csv(out / "baseline_dominance.csv", index=False)
    table = pd.DataFrame(paired)
    selected = table[table.recall_tolerance_pass & table.ndc_reduction.gt(0)].sort_values(["ef", "ndc"])
    report = (
        "# Actual online sweep\n\n"
        f"Recall tolerance: {args.recall_drop:.6f} absolute, or {100*args.recall_drop:.3f} percentage points.\n\n"
        "same_ef_vs_original.csv compares the SAME query IDs at the SAME ef. Standard errors use paired query differences.\n"
        "at_target_recall.csv selects the cheapest MEASURED point reaching each target, separately per config; "
        "unreached targets stay blank. pareto.csv includes ordinary baseline points.\n\n"
        "baseline.csv may contain additional ef points, on exactly the same query IDs. "
        "baseline_dominance.csv checks whether a measured baseline point has at least as much recall "
        "and at most as much NDC, with one strict improvement. Not dominated does not prove superiority.\n\n"
        "These runs retain diagnostic attribution, and boundtwohopcmp runs a baseline first. "
        "Latency is diagnostic; do not interpret it as a production speedup. NDC excludes evaluation distances.\n\n"
        "Candidates within tolerance and with lower measured NDC (not a final held-out selection):\n\n```text\n"
        + selected[["config", "ef", "recall", "ndc", "recall_delta", "ndc_reduction", "probe_reduction", "jump_reduction"]].to_string(index=False)
        + "\n```\n\nOrdinary baseline recall/NDC curve:\n\n```text\n"
        + baseline_points[["ef", "queries", "recall", "ndc"]].to_string(index=False)
        + f"\n```\n\nBaseline dominates {sum(r['dominated_by_baseline'] for r in dominance)}"
          f" / {len(dominance)} measured non-baseline points.\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
