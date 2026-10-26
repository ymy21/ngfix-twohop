#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def safe_ratio(a, b):
    return np.nan if b == 0 else a / b


def summarize_group(df, group_col, metrics):
    rows = []
    for name, g in df.groupby(group_col, dropna=False):
        row = {group_col: name, "events": len(g), "queries": g.query_id.nunique() if "query_id" in g else np.nan}
        for c in metrics:
            if c not in g.columns:
                continue
            x = pd.to_numeric(g[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if x.empty:
                continue
            row[f"{c}_mean"] = x.mean()
            row[f"{c}_median"] = x.median()
            row[f"{c}_p25"] = x.quantile(.25)
            row[f"{c}_p75"] = x.quantile(.75)
            row[f"{c}_p90"] = x.quantile(.90)
        rows.append(row)
    return pd.DataFrame(rows)


def node_reuse_table(df, cls_col="jump_class"):
    rows=[]
    for cls, g in df.groupby(cls_col):
        for role in ["cur_id", "mid_id", "nn_id"]:
            pn = g.groupby(role).agg(events=(role,"size"), queries=("query_id","nunique"))
            if pn.empty: continue
            repeated = pn.queries.ge(2)
            counts = pn.events.sort_values(ascending=False).to_numpy()
            total = int(counts.sum())
            n = len(counts)
            row={
                "jump_class":cls,"role":role.replace("_id",""),"events":total,"unique_nodes":n,
                "nodes_seen_by_2plus_queries":int(repeated.sum()),
                "share_events_on_nodes_seen_by_2plus_queries":safe_ratio(int(pn.loc[repeated,"events"].sum()),total),
                "max_queries_per_node":int(pn.queries.max()),
            }
            for frac in [.01,.05,.10]:
                k=max(1,int(np.ceil(n*frac)))
                row[f"top_{int(frac*100)}pct_nodes_event_coverage"]=safe_ratio(int(counts[:k].sum()),total)
            rows.append(row)
    return pd.DataFrame(rows)


def classify_mid(r):
    if not bool(r.get("success", 0)):
        return "FAILED_WITH_FRESH" if r.get("fresh_nn_cmps", 0) > 0 else "FAILED_NO_FRESH"
    if r.get("direct_new_gt", 0) > 0 or r.get("nearest_recovered_desc_gt", 0) > 0:
        return "SUCCESS_RECOVERED_GT"
    if r.get("direct_shared_gt", 0) > 0:
        return "SUCCESS_SHARED_GT_ONLY"
    return "SUCCESS_NO_GT_VALUE"


def matched_controls(mids):
    useful = mids[mids.mid_class.eq("SUCCESS_RECOVERED_GT")].copy()
    failed = mids[mids.mid_class.str.startswith("FAILED")].copy()
    if useful.empty or failed.empty:
        return pd.DataFrame()
    failed_groups = {k:g for k,g in failed.groupby(["query_id","first_edge_type"])}
    gap_scale = max(float(mids.mid_gap.quantile(.75)-mids.mid_gap.quantile(.25)), 1e-9)
    rows=[]
    for _, u in useful.iterrows():
        g = failed_groups.get((u.query_id,u.first_edge_type))
        if g is None or g.empty: continue
        hop_u = float(u.hop_over_ef) if pd.notna(u.hop_over_ef) else 0.0
        # Match on online-observable search state: same query, same first-edge type,
        # then nearest in hop/ef and mid_gap. GT is used only to label useful/failed.
        cost = (g.hop_over_ef.astype(float)-hop_u).abs()/0.05 + (g.mid_gap.astype(float)-float(u.mid_gap)).abs()/gap_scale
        j = cost.idxmin()
        f = g.loc[j]
        rows.append({
            "query_id":int(u.query_id),"useful_mid_id":int(u.mid_id),"failed_mid_id":int(f.mid_id),
            "first_edge_type":int(u.first_edge_type),"match_cost":float(cost.loc[j]),
            "useful_hop_over_ef":float(u.hop_over_ef),"failed_hop_over_ef":float(f.hop_over_ef),
            "useful_mid_gap":float(u.mid_gap),"failed_mid_gap":float(f.mid_gap),
        })
        for c in [
            "cur_total_degree","cur_repair_ratio","mid_total_degree","mid_repair_ratio",
            "cur_mid_data_dist","cur_mid_common_neighbors","cur_mid_jaccard",
            "best_nn_total_degree","best_nn_repair_ratio","mid_best_nn_data_dist",
            "mid_best_nn_common_neighbors","mid_best_nn_jaccard","cur_best_nn_data_dist",
            "cur_best_nn_common_neighbors","cur_best_nn_jaccard"
        ]:
            if c in mids.columns:
                rows[-1][f"useful_{c}"]=u[c]
                rows[-1][f"failed_{c}"]=f[c]
    return pd.DataFrame(rows)


def matched_summary(matched):
    if matched.empty: return matched
    rows=[]
    bases=sorted({c[len("useful_"):] for c in matched.columns if c.startswith("useful_") and f"failed_{c[len('useful_') :]}" in matched.columns})
    for c in bases:
        u=pd.to_numeric(matched[f"useful_{c}"],errors="coerce")
        f=pd.to_numeric(matched[f"failed_{c}"],errors="coerce")
        d=u-f
        rows.append({"feature":c,"useful_mean":u.mean(),"failed_mean":f.mean(),"mean_diff":d.mean(),
                     "useful_median":u.median(),"failed_median":f.median(),"median_diff":u.median()-f.median(),
                     "pairs":int((u.notna()&f.notna()).sum())})
    return pd.DataFrame(rows)


def kmeans_triplets(events, mids, outdir, k, sample_n):
    # This is motif-feature clustering, NOT whole-graph community detection.
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import MiniBatchKMeans
    except Exception as e:
        (outdir/"KMEANS_SKIPPED.txt").write_text(f"scikit-learn unavailable: {e}\n")
        return

    features=[
        "cur_dist","mid_dist","nn_dist","mid_gap","nn_margin","hop_over_ef",
        "cur_mid_data_dist","mid_nn_data_dist","cur_nn_data_dist",
        "cur_repair_ratio","mid_repair_ratio","nn_repair_ratio",
        "cur_mid_common_neighbors","mid_nn_common_neighbors","cur_nn_common_neighbors",
        "cur_mid_jaccard","mid_nn_jaccard","cur_nn_jaccard",
        "cur_total_degree","mid_total_degree","nn_total_degree",
        "rank_in_fresh_nn_by_query_dist"
    ]

    ev=events.copy()
    ev["analysis_class"]=ev.jump_class

    # Add failed-probe controls using the closest fresh nn scanned by each failed mid.
    # This gives the clustering a hard negative triplet with exactly the same
    # cur-mid-nn feature schema as admitted jump events.
    failed=mids[(~mids.success.astype(bool)) & mids.best_scored_nn.ge(0)].copy()
    if not failed.empty:
        fm=pd.DataFrame({
            "query_id":failed.query_id,
            "analysis_class":"FAILED_BEST_SCORED_NN",
            "cur_dist":failed.cur_dist,
            "mid_dist":failed.mid_dist,
            "nn_dist":failed.best_scored_nn_dist,
            "mid_gap":failed.mid_gap,
            "nn_margin":failed.bound_at_reject-failed.best_scored_nn_dist,
            "hop_over_ef":failed.hop_over_ef,
            "cur_mid_data_dist":failed.cur_mid_data_dist,
            "mid_nn_data_dist":failed.mid_best_nn_data_dist,
            "cur_nn_data_dist":failed.cur_best_nn_data_dist,
            "cur_repair_ratio":failed.cur_repair_ratio,
            "mid_repair_ratio":failed.mid_repair_ratio,
            "nn_repair_ratio":failed.best_nn_repair_ratio,
            "cur_mid_common_neighbors":failed.cur_mid_common_neighbors,
            "mid_nn_common_neighbors":failed.mid_best_nn_common_neighbors,
            "cur_nn_common_neighbors":failed.cur_best_nn_common_neighbors,
            "cur_mid_jaccard":failed.cur_mid_jaccard,
            "mid_nn_jaccard":failed.mid_best_nn_jaccard,
            "cur_nn_jaccard":failed.cur_best_nn_jaccard,
            "cur_total_degree":failed.cur_total_degree,
            "mid_total_degree":failed.mid_total_degree,
            "nn_total_degree":failed.best_nn_total_degree,
            "rank_in_fresh_nn_by_query_dist":1,
        })
        ev=pd.concat([ev, fm], ignore_index=True, sort=False)

    features=[c for c in features if c in ev.columns]
    work=ev[["query_id","analysis_class"]+features].replace([np.inf,-np.inf],np.nan).dropna()
    if len(work)>sample_n:
        # Stratified-ish sample so rare DIRECT_GT_NEW / ROUTER_TO_GT are not drowned
        # out completely by ROUTER_NO_GT / failed controls.
        parts=[]
        per=max(1000, sample_n//max(1,work.analysis_class.nunique()))
        for _,g in work.groupby("analysis_class"):
            parts.append(g.sample(min(len(g),per),random_state=0))
        work=pd.concat(parts,ignore_index=True)
        if len(work)>sample_n:
            work=work.sample(sample_n,random_state=0)
    if len(work)<k*10 or not features:
        (outdir/"KMEANS_SKIPPED.txt").write_text("Not enough complete rows/features for KMeans.\n")
        return
    X=StandardScaler().fit_transform(work[features].to_numpy(float))
    model=MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=min(4096,len(work)),n_init=10)
    work["cluster"]=model.fit_predict(X)
    comp=work.groupby(["cluster","analysis_class"]).size().rename("events").reset_index()
    comp["share_within_cluster"]=comp.events/comp.groupby("cluster").events.transform("sum")
    # Enrichment relative to the class base rate in the sampled clustering set.
    base=work.analysis_class.value_counts(normalize=True).rename("base_rate")
    comp=comp.merge(base,left_on="analysis_class",right_index=True,how="left")
    comp["enrichment_vs_base"]=comp.share_within_cluster/comp.base_rate
    comp.to_csv(outdir/"triplet_kmeans_cluster_composition.csv",index=False)
    cent=pd.DataFrame(model.cluster_centers_,columns=[f"z_{c}" for c in features])
    cent.insert(0,"cluster",range(k))
    cent.to_csv(outdir/"triplet_kmeans_standardized_centroids.csv",index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--events",required=True)
    ap.add_argument("--mids",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--kmeans",type=int,default=0,help="Optional number of motif clusters, e.g. 8. This is not graph community detection.")
    ap.add_argument("--kmeans-sample",type=int,default=200000)
    args=ap.parse_args()
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    events=pd.read_csv(args.events)
    mids=pd.read_csv(args.mids)
    if events.empty or mids.empty: raise RuntimeError("Empty trace CSV")

    # All admitted jump classes are kept separate.
    jump_metrics=[
        "hop_over_ef","cur_dist","mid_dist","nn_dist","mid_gap","nn_margin","rank_in_fresh_nn_by_query_dist",
        "cur_total_degree","cur_repair_degree","cur_repair_ratio",
        "mid_total_degree","mid_repair_degree","mid_repair_ratio",
        "nn_total_degree","nn_repair_degree","nn_repair_ratio",
        "cur_mid_data_dist","mid_nn_data_dist","cur_nn_data_dist",
        "cur_mid_common_neighbors","mid_nn_common_neighbors","cur_nn_common_neighbors",
        "cur_mid_jaccard","mid_nn_jaccard","cur_nn_jaccard",
    ]
    js=summarize_group(events,"jump_class",jump_metrics)
    # Add directly interpretable topology rates.
    rates=[]
    for cls,g in events.groupby("jump_class"):
        rates.append({
            "jump_class":cls,
            "cur_to_nn_direct_edge_rate":g.cur_to_nn_edge_type.ge(0).mean(),
            "nn_to_cur_direct_edge_rate":g.nn_to_cur_edge_type.ge(0).mean(),
            "mid_to_cur_reverse_edge_rate":g.mid_to_cur_edge_type.ge(0).mean(),
            "nn_to_mid_reverse_edge_rate":g.nn_to_mid_edge_type.ge(0).mean(),
            "cur_is_gt_rate":g.cur_is_gt.mean(),"mid_is_gt_rate":g.mid_is_gt.mean(),"nn_is_gt_rate":g.nn_is_gt.mean(),
            "baseline_scored_nn_rate":g.baseline_scored_nn.mean(),"baseline_expanded_nn_rate":g.baseline_expanded_nn.mean(),
        })
    js=js.merge(pd.DataFrame(rates),on="jump_class",how="left")
    js.to_csv(outdir/"jump_class_structural_summary.csv",index=False)
    node_reuse_table(events).to_csv(outdir/"node_reuse_by_jump_class.csv",index=False)

    # Mid-level categories include failed probes, so they answer MidGate questions.
    mids["mid_class"]=mids.apply(classify_mid,axis=1)
    mid_metrics=[
        "hop_over_ef","cur_dist","mid_dist","mid_gap","checked_nn","fresh_nn_cmps","admitted_nn",
        "cur_total_degree","cur_repair_degree","cur_repair_ratio",
        "mid_total_degree","mid_repair_degree","mid_repair_ratio",
        "cur_mid_data_dist","cur_mid_common_neighbors","cur_mid_jaccard",
        "best_scored_nn_dist","best_nn_total_degree","best_nn_repair_degree","best_nn_repair_ratio",
        "mid_best_nn_data_dist","mid_best_nn_common_neighbors","mid_best_nn_jaccard",
        "cur_best_nn_data_dist","cur_best_nn_common_neighbors","cur_best_nn_jaccard"
    ]
    summarize_group(mids,"mid_class",mid_metrics).to_csv(outdir/"mid_class_structural_summary.csv",index=False)

    # Best-scored NN is a hard control for failed mids: it is the closest fresh nn
    # found by that probe, even though the live beam admitted none.
    bsr=[]
    for cls,g in mids.groupby("mid_class"):
        g=g[g.best_scored_nn.ge(0)]
        if g.empty: continue
        bsr.append({
            "mid_class":cls,"mids":len(g),"best_scored_nn_is_gt_rate":g.best_scored_nn_is_gt.mean(),
            "best_scored_baseline_scored_rate":g.best_scored_baseline_scored.mean(),
            "best_scored_baseline_final_gt_rate":g.best_scored_baseline_final_gt.mean(),
            "best_scored_bound_final_gt_rate":g.best_scored_bound_final_gt.mean(),
            "best_scored_nn_dist_median":g.best_scored_nn_dist.median(),
            "best_scored_nn_repair_ratio_median":g.best_nn_repair_ratio.median(),
            "mid_best_nn_jaccard_median":g.mid_best_nn_jaccard.median(),
            "cur_best_nn_jaccard_median":g.cur_best_nn_jaccard.median(),
        })
    pd.DataFrame(bsr).to_csv(outdir/"best_scored_nn_by_mid_class.csv",index=False)

    matched=matched_controls(mids)
    matched.to_csv(outdir/"matched_useful_mid_vs_failed_pairs.csv",index=False)
    matched_summary(matched).to_csv(outdir/"matched_useful_mid_vs_failed_summary.csv",index=False)

    if args.kmeans>0:
        kmeans_triplets(events,mids,outdir,args.kmeans,args.kmeans_sample)

    print(f"events={len(events):,}, mids={len(mids):,}, queries={events.query_id.nunique():,}")
    print("\nJump classes:")
    print(events.jump_class.value_counts().to_string())
    print("\nMid classes:")
    print(mids.mid_class.value_counts().to_string())
    print(f"\nWrote structural analysis to {outdir}")

if __name__=="__main__":
    main()
