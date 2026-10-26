#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EDGE = {0: "H", 1: "N", 2: "R"}


def ratio(num, den):
    return np.nan if den == 0 else num / den


def concentration_table(df, node_col, query_col="query_id", label="nodes"):
    if df.empty:
        return pd.DataFrame(columns=["label", "metric", "value"])
    per_node = df.groupby(node_col).agg(events=(node_col, "size"), queries=(query_col, "nunique"))
    counts = per_node["events"].sort_values(ascending=False).to_numpy()
    n = len(counts)
    total = counts.sum()
    rows = [
        [label, "unique_nodes", n],
        [label, "events", total],
        [label, "nodes_seen_by_2plus_queries", int((per_node.queries >= 2).sum())],
        [label, "share_events_on_nodes_seen_by_2plus_queries",
         ratio(per_node.loc[per_node.queries >= 2, "events"].sum(), total)],
        [label, "max_queries_per_node", int(per_node.queries.max())],
    ]
    for frac in (0.001, 0.01, 0.05, 0.10):
        k = max(1, int(np.ceil(n * frac)))
        rows.append([label, f"top_{frac:g}_node_fraction_event_coverage", counts[:k].sum() / total])
    return pd.DataFrame(rows, columns=["label", "metric", "value"])


def split_half_overlap(df, node_col, useful_only=True, topk=1000):
    if useful_only:
        df = df[df["useful"].astype(bool)]
    if df.empty:
        return pd.DataFrame([{"topk": topk, "intersection": 0, "jaccard": np.nan}])
    qmax = int(df.query_id.max()) + 1
    a = df[df.query_id < qmax // 2].groupby(node_col).size().nlargest(topk).index
    b = df[df.query_id >= qmax // 2].groupby(node_col).size().nlargest(topk).index
    sa, sb = set(a), set(b)
    return pd.DataFrame([{
        "topk": topk,
        "intersection": len(sa & sb),
        "union": len(sa | sb),
        "jaccard": ratio(len(sa & sb), len(sa | sb)),
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="boundtwohopcmp main CSV")
    ap.add_argument("--events", required=True, help="*_jump_events.csv")
    ap.add_argument("--mids", required=True, help="*_jump_mids.csv")
    ap.add_argument("--hopbins", required=True, help="*_jump_hopbins.csv")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = pd.read_csv(args.result)
    result = result[result.row_type.eq("QUERY")].copy()
    events = pd.read_csv(args.events)
    mids = pd.read_csv(args.mids)
    hopbins = pd.read_csv(args.hopbins)

    if events.empty or mids.empty or hopbins.empty:
        raise RuntimeError("Trace CSV is empty; check --event_trace_ef and query limit.")

    trace_ef = int(events.ef.iloc[0])
    traced_qids = set(events.query_id.unique()) | set(mids.query_id.unique())
    result = result[result.ef.eq(trace_ef) & result.query_id.isin(traced_qids)].copy()

    # ------------------------------------------------------------------
    # 1) Candidate-level taxonomy and baseline visibility.
    # ------------------------------------------------------------------
    cls = events.groupby("jump_class").agg(
        events=("nn_id", "size"),
        queries=("query_id", "nunique"),
        baseline_scored_rate=("baseline_scored_nn", "mean"),
        baseline_expanded_rate=("baseline_expanded_nn", "mean"),
        hop_frac_mean=("hop_frac", "mean"),
        hop_frac_median=("hop_frac", "median"),
        mid_gap_mean=("mid_gap", "mean"),
        nn_margin_mean=("nn_margin", "mean"),
        recovered_desc_gt=("nearest_recovered_gt_descendants", "sum"),
    ).reset_index()
    cls["share_all_admitted_jumps"] = cls.events / cls.events.sum()
    cls.to_csv(outdir / "candidate_class_summary.csv", index=False)

    pair = events.assign(
        edge_pair=events.first_edge_type.map(EDGE) + events.second_edge_type.map(EDGE)
    ).groupby(["edge_pair", "jump_class"]).size().rename("events").reset_index()
    pair.to_csv(outdir / "candidate_class_by_edge_pair.csv", index=False)

    # Canonical edge-pair accounting.  These columns deliberately keep four
    # different denominators separate so percentages cannot be confused:
    # examined adjacency paths, fresh-scored nn, admitted jumps, and recovered
    # direct GT / recovered router descendants.
    pair_names = ["HH","HN","HR","NH","NN","NR","RH","RN","RR"]
    rq = result[result.ef.eq(trace_ef)]
    rows = []
    exam_total = sum(rq[f"exam_pair_{p}"].sum() for p in pair_names)
    score_total = sum(rq[f"score_pair_{p}"].sum() for p in pair_names)
    jump_total = sum(rq[f"jump_pair_{p}"].sum() for p in pair_names)
    for p in pair_names:
        exam = rq[f"exam_pair_{p}"].sum()
        score = rq[f"score_pair_{p}"].sum()
        jump = rq[f"jump_pair_{p}"].sum()
        pe = events[(events.first_edge_type.map(EDGE) + events.second_edge_type.map(EDGE)).eq(p)]
        direct_new = int(pe.jump_class.eq("DIRECT_GT_NEW").sum())
        recovered_desc = int(pe.nearest_recovered_gt_descendants.sum())
        rows.append({
            "edge_pair": p,
            "examined": exam,
            "fresh_scored": score,
            "admitted_jump": jump,
            "examined_share": ratio(exam, exam_total),
            "fresh_scored_share": ratio(score, score_total),
            "admitted_jump_share": ratio(jump, jump_total),
            "fresh_score_given_examined": ratio(score, exam),
            "admit_given_fresh_score": ratio(jump, score),
            "admit_given_examined": ratio(jump, exam),
            "direct_new_gt": direct_new,
            "recovered_router_desc_gt": recovered_desc,
        })
    edge_account = pd.DataFrame(rows)
    edge_account.to_csv(outdir / "edge_pair_accounting_9way.csv", index=False)

    # Optional H vs Repair(NGFix+RFix) collapse.  Exactly four composition rows
    # and their shares always sum to one within each denominator.
    def collapse_pair(p):
        a, b = p[0], p[1]
        return ("H" if a == "H" else "Repair") + "->" + ("H" if b == "H" else "Repair")
    edge4 = edge_account.assign(pair4=edge_account.edge_pair.map(collapse_pair)).groupby("pair4", as_index=False).agg(
        examined=("examined", "sum"), fresh_scored=("fresh_scored", "sum"),
        admitted_jump=("admitted_jump", "sum"), direct_new_gt=("direct_new_gt", "sum"),
        recovered_router_desc_gt=("recovered_router_desc_gt", "sum"))
    edge4["examined_share"] = edge4.examined / edge4.examined.sum()
    edge4["fresh_scored_share"] = edge4.fresh_scored / edge4.fresh_scored.sum()
    edge4["admitted_jump_share"] = edge4.admitted_jump / edge4.admitted_jump.sum()
    edge4["admit_given_fresh_score"] = edge4.admitted_jump / edge4.fresh_scored.replace(0, np.nan)
    edge4.to_csv(outdir / "edge_pair_accounting_4way_repair.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Timing: useful jumps vs all successful jumps.
    # ------------------------------------------------------------------
    events["useful"] = events["jump_class"].eq("DIRECT_GT_NEW") | events["nearest_recovered_gt_descendants"].gt(0)
    events["hop_decile"] = np.minimum(9, np.floor(events.hop_frac * 10).astype(int))
    cand_timing = events.groupby("hop_decile").agg(
        admitted_jumps=("nn_id", "size"),
        useful_jumps=("useful", "sum"),
        direct_new_gt=("jump_class", lambda s: (s == "DIRECT_GT_NEW").sum()),
        direct_shared_gt=("jump_class", lambda s: (s == "DIRECT_GT_SHARED").sum()),
        router_recovered=("nearest_recovered_gt_descendants", lambda s: (s > 0).sum()),
        baseline_scored_rate=("baseline_scored_nn", "mean"),
        mid_gap_mean=("mid_gap", "mean"),
        nn_margin_mean=("nn_margin", "mean"),
    ).reset_index()
    cand_timing["useful_share_of_admitted"] = cand_timing.useful_jumps / cand_timing.admitted_jumps
    cand_timing.to_csv(outdir / "candidate_timing_deciles.csv", index=False)

    # The mid-level denominator is more important: how profitable is probing at
    # each search stage, including failed mids and their actual fresh NDC cost?
    hb = hopbins.groupby(["bin", "hop_frac_lo", "hop_frac_hi"], as_index=False).agg(
        scans=("scans", "sum"),
        successful_mids=("successful_mids", "sum"),
        failed_mids=("failed_mids", "sum"),
        useful_mids=("useful_mids", "sum"),
        fresh_nn_ndc=("fresh_nn_ndc", "sum"),
        admitted_nn=("admitted_nn", "sum"),
        direct_new_gt=("direct_new_gt", "sum"),
        direct_shared_gt=("direct_shared_gt", "sum"),
        recovered_desc_gt=("recovered_desc_gt", "sum"),
        mid_gap_sum=("mid_gap_sum", "sum"),
        bound_sum=("bound_sum", "sum"),
        first_h=("first_h", "sum"),
        first_n=("first_n", "sum"),
        first_r=("first_r", "sum"),
    )
    hb["success_mid_rate"] = hb.successful_mids / hb.scans.replace(0, np.nan)
    hb["useful_mid_rate"] = hb.useful_mids / hb.scans.replace(0, np.nan)
    hb["useful_given_success"] = hb.useful_mids / hb.successful_mids.replace(0, np.nan)
    hb["incremental_gt"] = hb.direct_new_gt + hb.recovered_desc_gt
    hb["incremental_gt_per_1k_fresh_nn_ndc"] = 1000 * hb.incremental_gt / hb.fresh_nn_ndc.replace(0, np.nan)
    hb["shared_gt_per_1k_fresh_nn_ndc"] = 1000 * hb.direct_shared_gt / hb.fresh_nn_ndc.replace(0, np.nan)
    hb["avg_mid_gap"] = hb.mid_gap_sum / hb.scans.replace(0, np.nan)
    hb["avg_bound"] = hb.bound_sum / hb.scans.replace(0, np.nan)
    hb.to_csv(outdir / "mid_probe_timing_20bins.csv", index=False)

    # First successful/useful probe per query.
    first_success = mids[mids.success.astype(bool)].groupby("query_id")["hop_frac"].min().rename("first_success_hop_frac")
    first_useful = mids[mids.useful_mid.astype(bool)].groupby("query_id")["hop_frac"].min().rename("first_useful_hop_frac")
    first = pd.concat([first_success, first_useful], axis=1)
    first.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).to_csv(outdir / "first_jump_timing_quantiles.csv")

    # Early/late coarse comparison for the research decision.
    hb2 = hopbins.copy()
    hb2["phase"] = pd.cut(
        hb2.hop_frac_lo,
        bins=[-1e-9, .25, .50, .75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
        include_lowest=True,
        right=False,
    )
    phase = hb2.groupby("phase", observed=True).agg(
        scans=("scans", "sum"), successful_mids=("successful_mids", "sum"),
        useful_mids=("useful_mids", "sum"), fresh_nn_ndc=("fresh_nn_ndc", "sum"),
        direct_new_gt=("direct_new_gt", "sum"), direct_shared_gt=("direct_shared_gt", "sum"),
        recovered_desc_gt=("recovered_desc_gt", "sum")
    ).reset_index()
    phase["success_mid_rate"] = phase.successful_mids / phase.scans.replace(0, np.nan)
    phase["useful_mid_rate"] = phase.useful_mids / phase.scans.replace(0, np.nan)
    phase["incremental_gt"] = phase.direct_new_gt + phase.recovered_desc_gt
    phase["incremental_gt_per_1k_fresh_nn_ndc"] = 1000 * phase.incremental_gt / phase.fresh_nn_ndc.replace(0, np.nan)
    phase.to_csv(outdir / "early_late_phase_summary.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Cross-query concentration / "jumpable region" evidence.
    # ------------------------------------------------------------------
    useful_events = events[events.useful]
    shared_events = events[events["jump_class"].eq("DIRECT_GT_SHARED")]
    useful_mids = mids[mids.useful_mid.astype(bool)]
    successful_mids = mids[mids.success.astype(bool)]

    conc = pd.concat([
        concentration_table(successful_mids, "mid_id", label="all_successful_mid"),
        concentration_table(useful_mids, "mid_id", label="useful_mid"),
        concentration_table(useful_events, "cur_id", label="useful_cur"),
        concentration_table(useful_events, "nn_id", label="useful_nn"),
        concentration_table(shared_events, "nn_id", label="shared_direct_gt_nn"),
    ], ignore_index=True)
    conc.to_csv(outdir / "node_concentration.csv", index=False)

    split_half_overlap(events, "mid_id", useful_only=True, topk=100).to_csv(outdir / "useful_mid_split_half_top100.csv", index=False)
    split_half_overlap(events, "mid_id", useful_only=True, topk=1000).to_csv(outdir / "useful_mid_split_half_top1000.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Query-level interaction: easy (baseline recall=1) vs non-perfect.
    # ------------------------------------------------------------------
    perq_mid = mids.groupby("query_id").agg(
        scans=("mid_id", "size"),
        successful_mids=("success", "sum"),
        useful_mids=("useful_mid", "sum"),
        fresh_nn_ndc=("fresh_nn_cmps", "sum"),
        direct_new_gt=("direct_new_gt", "sum"),
        direct_shared_gt=("direct_shared_gt", "sum"),
        recovered_desc_gt=("nearest_recovered_desc_gt", "sum"),
    ).reset_index()
    qr = result.merge(perq_mid, on="query_id", how="inner")
    qr["query_group"] = np.where(np.isclose(qr.paired_baseline_recall, 1.0), "baseline_recall_1", "baseline_recall_lt_1")
    qgrp = qr.groupby("query_group").agg(
        queries=("query_id", "size"),
        baseline_recall=("paired_baseline_recall", "mean"),
        bound_recall=("recall", "mean"),
        bound_ndc=("ndc", "mean"),
        bound_mid_ndc=("ndc_mid", "mean"),
        bound_nn_ndc=("ndc_nn", "mean"),
        scans=("scans", "mean"),
        successful_mids=("successful_mids", "mean"),
        useful_mids=("useful_mids", "mean"),
        direct_new_gt=("direct_new_gt", "mean"),
        direct_shared_gt=("direct_shared_gt", "mean"),
        recovered_desc_gt=("recovered_desc_gt", "mean"),
    ).reset_index()
    if "skipped_mid_from_success_probe" in qr.columns:
        extra = qr.groupby("query_group").agg(
            unique_firsthop_nodes_pre_scored_by_success_probe=("skipped_mid_from_success_probe", "mean"),
            unique_firsthop_nodes_pre_scored_by_failed_probe=("skipped_mid_from_failed_probe", "mean"),
            unique_firsthop_nodes_pre_scored_as_admitted_jump=("skipped_mid_from_admitted_jump", "mean"),
        ).reset_index()
        qgrp = qgrp.merge(extra, on="query_group", how="left")
    qgrp.to_csv(outdir / "query_group_easy_vs_nonperfect.csv", index=False)

    # Human-readable console summary.
    print(f"trace ef={trace_ef}, queries={events.query_id.nunique()}")
    print("\nCandidate classes:")
    print(cls[["jump_class", "events", "share_all_admitted_jumps", "baseline_scored_rate", "hop_frac_median"]].to_string(index=False))
    print("\nEarly/late probe efficiency:")
    print(phase[["phase", "scans", "success_mid_rate", "useful_mid_rate", "incremental_gt_per_1k_fresh_nn_ndc"]].to_string(index=False))
    print("\nQuery groups:")
    print(qgrp.to_string(index=False))
    print(f"\nWrote analysis to {outdir}")


if __name__ == "__main__":
    main()
