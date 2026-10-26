"""Read existing per-query measurements; test residual-error and tail hypotheses.

No new search runs. Grouping by GT recall and oracle choices are offline diagnostics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'result/complete_experiment_report_20261013/tables'
OUT=ROOT/'result/residual_query_analysis_20261014'
OUT.mkdir(exist_ok=True)
q=pd.read_csv(SOURCE/'performance_queries.csv')
l=pd.read_csv(SOURCE/'performance_latencies.csv')
p=pd.read_csv(SOURCE/'performance_all_points.csv')
data=q.merge(l.drop(columns=['ndc']),on=['graph','k','method','ef','query_id'],validate='one_to_one')
assert len(data)==144000 and data.groupby(['graph','k','method','ef']).size().eq(1000).all()

def frame(graph,k,method,ef):
    s=data[(data.graph==graph)&(data.k==k)&(data.method==method)&(data.ef==ef)]
    return s.set_index('query_id').sort_index()[['recall','ndc','latency_us']]

def compare(a,b,mask=None):
    assert a.index.equals(b.index)
    if mask is not None:a=a.loc[mask];b=b.loc[mask]
    dr=a.recall-b.recall
    return dict(queries=len(a),a_recall=a.recall.mean(),b_recall=b.recall.mean(),delta_recall=dr.mean(),
        better_recall=int((dr>1e-7).sum()),equal_recall=int((dr.abs()<=1e-7).sum()),worse_recall=int((dr< -1e-7).sum()),
        a_ndc=a.ndc.mean(),b_ndc=b.ndc.mean(),a_latency_mean_us=a.latency_us.mean(),b_latency_mean_us=b.latency_us.mean(),
        a_p95_us=a.latency_us.quantile(.95),b_p95_us=b.latency_us.quantile(.95),
        better_recall_and_no_more_ndc=int(((dr>1e-7)&(a.ndc<=b.ndc)).sum()),
        better_recall_and_no_more_latency=int(((dr>1e-7)&(a.latency_us<=b.latency_us)).sum()),
        better_recall_and_both_costs=int(((dr>1e-7)&(a.ndc<=b.ndc)&(a.latency_us<=b.latency_us)).sum()),
        oracle_choose_by_gt_recall=np.maximum(a.recall,b.recall).mean(),
        win_recall_gain=dr.clip(lower=0).mean(),loss_recall_gap=(-dr).clip(lower=0).mean())

matched=[];same=[];tails=[];deciles=[];residual=[]
for graph in ['hnsw','ngfix']:
  for k in [10,100]:
    candidates=p[(p.graph==graph)&(p.k==k)&(p.method=='bound_unbounded')]
    for _,r in candidates.iterrows():
      for baseline_graph in ([graph,'ngfix'] if graph=='hnsw' else [graph]):
        beams=p[(p.graph==baseline_graph)&(p.k==k)&(p.method=='beam')]
        eligible=beams[beams.recall>=r.recall-1e-12]
        if eligible.empty:
          matched.append(dict(graph=graph,k=k,a_ef=int(r.ef),b_graph=baseline_graph,comparable=False));continue
        best=eligible.loc[eligible.qps_median.idxmax()]
        a=frame(graph,k,'bound_unbounded',r.ef);b=frame(baseline_graph,k,'beam',best.ef)
        matched.append(dict(graph=graph,k=k,a_ef=int(r.ef),b_graph=baseline_graph,b_ef=int(best.ef),comparable=True,
            beam_qps_speedup=best.qps_median/r.qps_median,beam_ndc_speedup=r.mean_ndc/best.mean_ndc,
            beam_p95_speedup=r.latency_p95_us/best.latency_p95_us,**compare(a,b)))
      a=frame(graph,k,'bound_unbounded',r.ef);b=frame('ngfix',k,'beam',r.ef)
      info=dict(a_graph=graph,k=k,a_ef=int(r.ef),b_graph='ngfix',b_ef=int(r.ef))
      same.append(dict(**info,**compare(a,b)))
      ranked=b.reset_index().sort_values(['recall','ndc','query_id'],ascending=[True,False,True])
      labels={int(row.query_id):i//100 for i,(_,row) in enumerate(ranked.iterrows())}
      for label,mask in [('all',np.ones(len(b),bool)),('ngfix_lowest_recall_20pct',b.index.isin(ranked.head(200).query_id)),
                         ('remaining_80pct',~b.index.isin(ranked.head(200).query_id)),('ngfix_not_perfect',b.recall<1-1e-7),
                         ('ngfix_below_95pct',b.recall<.95-1e-7),('ngfix_below_90pct',b.recall<.9-1e-7)]:
        if np.any(mask):tails.append(dict(**info,group=label,**compare(a,b,mask)))
      for decile in range(10):
        mask=b.index.isin([qid for qid,group in labels.items() if group==decile])
        deciles.append(dict(**info,decile_hardest_first=decile+1,**compare(a,b,mask)))
      if graph=='ngfix':
        missing=np.maximum(0,1-b.recall)*k
        gain=(a.recall-b.recall)*k
        residual.append(dict(k=k,ef=int(r.ef),baseline_missing_gt=missing.sum(),net_additional_gt=gain.sum(),
            net_missing_fraction_closed=gain.sum()/missing.sum() if missing.sum() else np.nan,
            extra_ndc=(a.ndc-b.ndc).sum(),extra_us=(a.latency_us-b.latency_us).sum(),
            extra_ndc_per_net_gt=(a.ndc-b.ndc).sum()/gain.sum() if gain.sum()>0 else np.nan))

for name,rows in [('matched_recall',matched),('same_ef_vs_ngfix',same),('ngfix_fixed_groups',tails),
                  ('ngfix_recall_deciles',deciles),('ngfix_residual_error_closed',residual)]:
    pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)

# Anchor fixed NGFix difficulty across ALL methods and ef values.
anchors=[]
for k,anchor_ef in [(10,100),(100,400)]:
    b=frame('ngfix',k,'beam',anchor_ef)
    ranked=b.reset_index().sort_values(['recall','ndc','query_id'],ascending=[True,False,True])
    ids=set(ranked.head(200).query_id)
    for keys,s in data[data.k==k].groupby(['graph','method','ef']):
        a=s.set_index('query_id').sort_index()[['recall','ndc','latency_us']]
        for label,mask in [('ngfix_anchor_lowest_recall_20pct',a.index.isin(ids)),('remaining_80pct',~a.index.isin(ids))]:
            anchors.append(dict(graph=keys[0],method=keys[1],ef=keys[2],k=k,anchor_ef=anchor_ef,group=label,**compare(a,b,mask)))
pd.DataFrame(anchors).to_csv(OUT/'fixed_ngfix_anchor_groups.csv',index=False)

# In the specific hardest decile, check whether widening NGFix beam already
# obtains the apparent same-ef HNSW-twohop advantage at lower cost.
tail_matches=[];tail_curves=[]
for k,anchor_ef in [(10,10),(10,15),(10,20),(10,100),(100,100),(100,400)]:
    b=frame('ngfix',k,'beam',anchor_ef)
    rank=b.reset_index().sort_values(['recall','ndc','query_id'],ascending=[True,False,True])
    for share in [.1,.2]:
      ids=set(rank.head(int(1000*share)).query_id)
      mask=b.index.isin(ids)
      a=frame('hnsw',k,'bound_unbounded',anchor_ef).loc[mask]
      rows=[]
      for _,r in p[(p.graph=='ngfix')&(p.k==k)&(p.method=='beam')].iterrows():
        c=frame('ngfix',k,'beam',r.ef).loc[mask]
        z=dict(k=k,anchor_ef=anchor_ef,share=share,beam_ef=int(r.ef),beam_recall=c.recall.mean(),
               beam_ndc=c.ndc.mean(),beam_us=c.latency_us.mean(),beam_p95_us=c.latency_us.quantile(.95),
               bound_recall=a.recall.mean(),bound_ndc=a.ndc.mean(),bound_us=a.latency_us.mean(),
               bound_p95_us=a.latency_us.quantile(.95))
        rows.append(z);tail_curves.append(z)
      eligible=[r for r in rows if r['beam_recall']>=a.recall.mean()-1e-12]
      if eligible:
        best=min(eligible,key=lambda r:r['beam_us']);tail_matches.append(best)
pd.DataFrame(tail_matches).to_csv(OUT/'ngfix_tail_matched_recall.csv',index=False)
pd.DataFrame(tail_curves).to_csv(OUT/'ngfix_tail_beam_sweeps.csv',index=False)

# Mechanism GT counts are exact and include identities via recorded attribution.
m=pd.read_csv(SOURCE/'mechanism_cases_enriched.csv')
mech=[]
for _,r in m.iterrows():
    missing=(1-r.paired_baseline_recall_mean)*r.k*1000
    net=(r.recovered_gt_total_mean-r.lost_gt_total_mean)*1000
    mech.append(dict(case=r['case'],missing_gt=missing,recovered_gt=r.recovered_gt_total_mean*1000,
                     lost_gt=r.lost_gt_total_mean*1000,net_gt=net,net_fraction_of_missing_closed=net/missing,
                     extra_ndc_per_net_gt=r.delta_ndc*1000/net))
pd.DataFrame(mech).to_csv(OUT/'mechanism_missing_error_closed.csv',index=False)
df=pd.DataFrame(matched);comp=df[df.comparable]
summary=dict(source_rows=len(data),matched_comparisons=len(comp),unmatched=int((~df.comparable).sum()),
             beam_faster=int((comp.beam_qps_speedup>1).sum()),beam_less_ndc=int((comp.beam_ndc_speedup>1).sum()),
             beam_better_p95=int((comp.beam_p95_speedup>1).sum()),
             input_note='first 1000 test queries, one performance round, no test-train coverage distances available')
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\nSame-ef selected configurations versus NGFix beam (A=bound):')
print(pd.DataFrame(same).query('(k==10 and a_ef in [20,100]) or (k==100 and a_ef in [100,400])')
      [['a_graph','k','a_ef','a_recall','b_recall','better_recall','worse_recall','better_recall_and_no_more_ndc','a_ndc','b_ndc','a_p95_us','b_p95_us']].to_string(index=False))
print('\nLowest-recall 20% selected configurations:')
print(pd.DataFrame(tails).query('group=="ngfix_lowest_recall_20pct" and ((k==10 and a_ef in [20,100]) or (k==100 and a_ef in [100,400]))')
      [['a_graph','k','a_ef','a_recall','b_recall','better_recall','equal_recall','worse_recall','a_ndc','b_ndc','a_latency_mean_us','b_latency_mean_us']].to_string(index=False))
print('\nExact recorded mechanism missed-GT closure:')
print(pd.DataFrame(mech).to_string(index=False))
print('\nMatched recall within fixed NGFix-hard cohorts:')
print(pd.DataFrame(tail_matches).to_string(index=False))
