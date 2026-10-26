#!/usr/bin/env python3
import argparse
import pandas as pd

PAIR_NAMES = ["HH","HN","HR","NH","NN","NR","RH","RN","RR"]

def load_query_rows(path):
    df = pd.read_csv(path)
    return df[df["row_type"] == "QUERY"].copy()

def summarize(bound_path, baseline_path=None):
    b = load_query_rows(bound_path)
    base = load_query_rows(baseline_path) if baseline_path else None

    rows = []
    for ef, g in b.groupby("ef"):
        row = {
            "ef": int(ef),
            "recall": g.recall.mean(),
            "ndc": g.ndc.mean(),
            "twohop_mid_scans": g.twohop_mid_scans.mean(),
            "jump_mids": g.jump_mids.mean(),
            "jump_mid_success_rate": g.jump_mids.sum() / max(g.twohop_mid_scans.sum(), 1),
            "jump_candidates": g.jump_candidates.mean(),
            "jump_candidates_gt": g.jump_candidates_gt.mean(),
            "jump_gt_rate": g.jump_candidates_gt.sum() / max(g.jump_candidates.sum(), 1),
            "final_gt_direct_jump": g.final_gt_direct_jump.mean(),
            "final_gt_bridge_descendant": g.final_gt_bridge_descendant.mean(),
            "final_gt_non_gt_bridge_root": g.final_gt_non_gt_bridge_root.mean(),
            "final_topk_bridge_descendant": g.final_topk_bridge_descendant.mean(),
        }
        if "recovered_gt_total" in g.columns:
            row["recovered_gt_total"] = g.recovered_gt_total.mean()
            row["recovered_gt_direct_jump"] = g.recovered_gt_direct_jump.mean()
            row["recovered_gt_bridge_descendant"] = g.recovered_gt_bridge_descendant.mean()
            row["recovered_gt_non_gt_bridge_root"] = g.recovered_gt_non_gt_bridge_root.mean()
            row["lost_gt_total"] = g.lost_gt_total.mean()
            denom_rec = max(g.recovered_gt_total.sum(), 1)
            row["recovered_direct_share"] = g.recovered_gt_direct_jump.sum() / denom_rec
            row["recovered_bridge_desc_share"] = g.recovered_gt_bridge_descendant.sum() / denom_rec
            row["recovered_non_gt_router_share"] = g.recovered_gt_non_gt_bridge_root.sum() / denom_rec
        for p in PAIR_NAMES:
            exam = g[f"exam_pair_{p}"].sum()
            jump = g[f"jump_pair_{p}"].sum()
            row[f"yield_{p}"] = jump / max(exam, 1)
            row[f"jump_share_{p}"] = jump / max(sum(g[f"jump_pair_{x}"].sum() for x in PAIR_NAMES), 1)
        if base is not None:
            gb = base[base.ef == ef].set_index("query_id")
            gg = g.set_index("query_id")
            common = gg.index.intersection(gb.index)
            delta_r = gg.loc[common, "recall"] - gb.loc[common, "recall"]
            delta_n = gg.loc[common, "ndc"] - gb.loc[common, "ndc"]
            row["delta_recall"] = delta_r.mean()
            row["delta_ndc"] = delta_n.mean()
            row["queries_recall_up_frac"] = (delta_r > 0).mean()
            row["queries_recall_down_frac"] = (delta_r < 0).mean()
            row["delta_recall_per_1k_ndc"] = delta_r.mean() / max(delta_n.mean() / 1000.0, 1e-12)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("ef")
    print(out.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound", required=True)
    ap.add_argument("--baseline")
    args = ap.parse_args()
    summarize(args.bound, args.baseline)
