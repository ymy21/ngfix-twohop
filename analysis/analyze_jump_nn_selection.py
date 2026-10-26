#!/usr/bin/env python3
"""Offline NN rank retention is diagnostic; online sweeps alone measure recall/NDC."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize_jump_gates import load_run


def diagnose(frame, limit=8):
    """Only uncensored, fully exact-scored pools have identifiable rank labels."""
    rows=[]; census=[]
    for (qid,ef,pid), g in frame.groupby(['query_id','ef','probe_id'],sort=True):
        if g.nn_id.duplicated().any(): raise ValueError('Duplicate NN in a probe')
        complete=(not g.censored.any() and g.scored.eq(1).all() and
                  g.pool_size.eq(len(g)).all() and g.scored_size.eq(len(g)).all())
        census.append(dict(query_id=qid,ef=ef,probe_id=pid,pool_size=len(g),complete=complete,
                           first_type=int(g.first_type.iloc[0]),hop=int(g.hop.iloc[0])))
        if not complete: continue
        if not np.isfinite(g[['exact_dist','cheap_score']]).all().all(): raise ValueError('Nonfinite scored distance')
        if not g.value_label.isin([0,1]).all(): raise ValueError('Invalid complete-pool labels')
        n=min(limit,len(g))
        selectors={
            'prefix':g.sort_values('adjacency_ordinal').head(n),
            'random':g.sort_values(['random_key','adjacency_ordinal']).head(n),
            'cheap':g.sort_values(['cheap_score','adjacency_ordinal']).head(n),
            'oracle_distance':g.sort_values(['exact_dist','adjacency_ordinal']).head(n)}
        best=g.sort_values(['exact_dist','adjacency_ordinal']).iloc[0].nn_id
        top_exact=set(selectors['oracle_distance'].nn_id)
        for mode,selected in selectors.items():
            rows.append(dict(query_id=qid,ef=ef,probe_id=pid,selector=mode,
                pool_size=len(g),selected=len(selected),first_type=int(g.first_type.iloc[0]),
                hop=int(g.hop.iloc[0]),value_total=int(g.value_label.sum()),
                value_kept=int(selected.value_label.sum()),
                direct_total=int(g.direct_new_gt.sum()),direct_kept=int(selected.direct_new_gt.sum()),
                router_total=int(g.nearest_recovered_desc_gt.gt(0).sum()),
                router_kept=int(selected.nearest_recovered_desc_gt.gt(0).sum()),
                descendant_total=int(g.nearest_recovered_desc_gt.sum()),
                descendant_kept=int(selected.nearest_recovered_desc_gt.sum()),
                best_exact_kept=int(best in set(selected.nn_id)),
                top_exact_total=n,top_exact_kept=len(top_exact & set(selected.nn_id))))
    return pd.DataFrame(rows),pd.DataFrame(census)


def retention_summary(rows, extra=()):
    summaries=[]
    for keys,g in rows.groupby(['ef','selector']+list(extra),sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(['ef','selector']+list(extra),keys))
        row.update(probes=len(g),queries=g.query_id.nunique(),
                   useful_probes=int(g.value_total.gt(0).sum()),
                   useful_queries=g.loc[g.value_total.gt(0),'query_id'].nunique(),
                   exact_fraction=g.selected.sum()/g.pool_size.sum(),best_exact_retention=g.best_exact_kept.mean())
        for metric in ('value','direct','router','descendant','top_exact'):
            total=g[metric+'_total'].sum(); kept=g[metric+'_kept'].sum()
            row.update({metric+'_total':total,metric+'_kept':kept,
                        metric+'_retention':kept/total if total else np.nan})
        summaries.append(row)
    return pd.DataFrame(summaries)


def endpoint_type_summary(frame, limit):
    records=[]
    for (_,ef,_),g in frame.groupby(['query_id','ef','probe_id']):
        if g.censored.any() or not g.scored.eq(1).all() or not g.pool_size.eq(len(g)).all(): continue
        n=min(limit,len(g))
        orders={'prefix':['adjacency_ordinal'],'random':['random_key','adjacency_ordinal'],
                'cheap':['cheap_score','adjacency_ordinal'],'oracle_distance':['exact_dist','adjacency_ordinal']}
        for mode,order in orders.items():
            selected=set(g.sort_values(order).head(n).nn_id)
            for edge,part in g.groupby('second_type'):
                kept=part[part.nn_id.isin(selected)]
                records.append(dict(ef=ef,selector=mode,second_type=edge,pool_targets=len(part),
                    selected_targets=len(kept),value_total=part.value_label.sum(),value_kept=kept.value_label.sum()))
    if not records: return pd.DataFrame()
    out=pd.DataFrame(records).groupby(['ef','selector','second_type'],as_index=False).sum()
    out['value_retention']=out.value_kept/out.value_total.replace(0,np.nan)
    out['selected_fraction']=out.selected_targets/out.pool_targets
    return out


def diagnostic(a,m,out):
    parts=[pd.read_csv(path,dtype={'random_key':'uint64'}) for path in a.indir.glob('*_nn_candidates.csv')]
    if not parts: raise ValueError('No NN traces')
    raw=pd.concat(parts,ignore_index=True)
    expected=list(range(m['controls_settings']['offset'],m['controls_settings']['offset']+m['controls_settings']['limit']))
    coverage=[]
    for rec in m['cases']:
        run=load_run(a.indir/(rec['name']+'.csv'))
        for ef,g in run.groupby(level='ef'):
            if g.index.get_level_values('query_id').tolist()!=expected: raise ValueError('Query IDs differ')
            events=raw[raw.ef.eq(ef)]
            counts=events[events.scored.eq(1)].groupby('query_id').size().reindex(expected,fill_value=0)
            if not np.array_equal(counts.to_numpy(),g.ndc_nn.to_numpy()):
                raise ValueError('Trace scored rows do not equal measured NN NDC')
            coverage.append(dict(ef=ef,queries=len(g),trace_queries=events.query_id.nunique(),
                zero_nn_queries=int(g.ndc_nn.eq(0).sum()),total_probes=int(g.twohop_mid_scans.sum()),
                nonempty_pools=events[['query_id','probe_id']].drop_duplicates().shape[0],
                nn_ndc=g.ndc_nn.mean(),ndc=g.ndc.mean(),recall=g.recall.mean()))
    pd.DataFrame(coverage).to_csv(out/'diagnostic_run_summary.csv',index=False)
    rows,census=diagnose(raw,m['parameters']['nn_limit'])
    census.to_csv(out/'pool_census.csv',index=False)
    rows.to_csv(out/'offline_probe_retention.csv',index=False)
    if rows.empty:
        print('No complete pools: lower diagnostic cost or run with NN_CAP=0. No retention conclusion.'); return
    summary=retention_summary(rows)
    summary.to_csv(out/'offline_nn_retention.csv',index=False)
    retention_summary(rows,['first_type']).to_csv(out/'offline_by_mid_type.csv',index=False)
    endpoint_type_summary(raw,m['parameters']['nn_limit']).to_csv(out/'offline_by_nn_type.csv',index=False)
    counts=census.groupby('ef').agg(probes=('probe_id','size'),complete=('complete','sum'))
    counts['complete_fraction']=counts.complete/counts.probes
    counts.to_csv(out/'pool_coverage.csv')
    # Query-cluster bootstrap, not independent-NN uncertainty.
    rng=np.random.default_rng(20261008); ci=[]
    for ef,group in rows.groupby('ef'):
        ids=np.array(expected)
        for reference in ('prefix','random','oracle_distance'):
            tables=[]
            for mode in ('cheap',reference):
                tables.append(group[group.selector.eq(mode)].groupby('query_id')[['value_kept','value_total']].sum().reindex(ids,fill_value=0).to_numpy())
            if not tables[0][:,1].sum(): continue
            values=[]
            for _ in range(2000):
                ix=rng.integers(0,len(ids),len(ids)); totals=[t[ix].sum(axis=0) for t in tables]
                if totals[0][1]: values.append(totals[0][0]/totals[0][1]-totals[1][0]/totals[1][1])
            ci.append(dict(ef=ef,comparison='cheap_minus_'+reference,
                delta=tables[0][:,0].sum()/tables[0][:,1].sum()-tables[1][:,0].sum()/tables[1][:,1].sum(),
                ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975),
                useful_queries=int((tables[0][:,1]>0).sum())))
    pd.DataFrame(ci).to_csv(out/'offline_query_bootstrap.csv',index=False)
    print(counts.to_string()); print(summary.to_string(index=False))
    print('Retention is conditional on FULL trajectories and complete pools; it is not online recall.')


def online(a,m,out):
    names=[r['name'] for r in m['cases'] if r['name']!='dabs']
    runs={name:load_run(a.indir/(name+'.csv')) for name in names}
    ref=runs['nn_full']; baseline=runs['baseline']; pairs=[]; summaries=[]; query_deltas=[]
    p=m['parameters']; dim=m['dimension']
    expected=list(range(p['offset'],p['offset']+p['limit']))
    for name,f in runs.items():
        for ef,g in f.groupby(level='ef'):
            if g.index.get_level_values('query_id').tolist()!=expected: raise ValueError('Query IDs differ')
        if name!='baseline':
            if not f.index.equals(ref.index): raise ValueError('Different query/ef pairs')
            b=baseline.loc[f.index]
            if not np.allclose(f.paired_baseline_recall,b.recall) or not np.array_equal(f.paired_baseline_ndc,b.ndc):
                raise ValueError('Paired beam differs')
            if p['nn_cap'] and f.ndc_nn.gt(p['nn_cap']).any(): raise ValueError('Query NN cap exceeded')
            if name!='nn_full' and f.ndc_nn.gt(p['nn_limit']*f.twohop_mid_scans).any():
                raise ValueError('Per-mid selector cap exceeded')
            if f.gate_nn_shadow_scores.ne(0).any(): raise ValueError('Online ablation must disable shadow scoring')
            if name.startswith('nn_cheap'):
                d=int(name.rsplit('_d',1)[1])
                if not np.array_equal(f.gate_nn_cheap_products,d*f.gate_nn_cheap_scores):
                    raise ValueError('Cheap product accounting mismatch')
            elif f.gate_nn_cheap_scores.ne(0).any(): raise ValueError('Unexpected cheap scoring')
        for ef,g in f.groupby(level='ef'):
            summary=dict(config=name,ef=ef,queries=len(g))
            for col in ['recall','ndc','ndc_mid','ndc_nn','twohop_mid_scans','latency_us',
                        'gate_nn_pool_targets','gate_nn_selected_targets','gate_nn_cheap_scores',
                        'gate_nn_cheap_products','gate_nn_query_dims_inspected']:
                summary[col]=g[col].mean()
            summary['distance_equivalent_products']=summary['ndc']+summary['gate_nn_cheap_products']/dim
            summary['nn_budget_full_fraction']=g.ndc_nn.eq(p['nn_cap']).mean() if p['nn_cap'] else np.nan
            summaries.append(summary)
            if name=='baseline': continue
            control=ref.loc[g.index]; dr=g.recall-control.recall; dn=g.ndc-control.ndc
            pairs.append(dict(config=name,ef=ef,queries=len(g),recall_delta_pp=100*dr.mean(),
                recall_delta_ci_low_pp=100*(dr.mean()-1.96*dr.std(ddof=1)/np.sqrt(len(g))),
                recall_delta_ci_high_pp=100*(dr.mean()+1.96*dr.std(ddof=1)/np.sqrt(len(g))),
                ndc_delta=dn.mean(),ndc_delta_se=dn.std(ddof=1)/np.sqrt(len(g)),
                mid_ndc_delta=(g.ndc_mid-control.ndc_mid).mean(),
                nn_ndc_delta=(g.ndc_nn-control.ndc_nn).mean(),
                recall_drop_queries=int(dr.lt(0).sum()),recall_gain_queries=int(dr.gt(0).sum())))
            query_deltas += [dict(config=name,ef=ef,query_id=key[1],recall_delta_pp=100*dr.loc[key],ndc_delta=dn.loc[key]) for key in g.index]
    points=pd.DataFrame(summaries)
    dpath=a.indir/'dabs/dabs.csv'
    if dpath.exists():
        df=pd.read_csv(dpath); df=df[df.row_type.eq('QUERY')]
        for gamma,g in df.groupby('dabs_gamma'):
            if g.query_id.sort_values().tolist()!=expected: raise ValueError('DABS query IDs differ')
            if not np.allclose(g.ndc,g.ndc_entry+g.ndc_mid+g.ndc_nn): raise ValueError('DABS NDC mismatch')
            points.loc[len(points)]=dict(config=f'dabs_gamma{gamma:g}',ef=0,queries=len(g),
                recall=g.recall.mean(),ndc=g.ndc.mean(),ndc_mid=g.ndc_mid.mean(),ndc_nn=g.ndc_nn.mean(),
                latency_us=g.latency_us.mean(),distance_equivalent_products=g.ndc.mean())
    points.to_csv(out/'online_summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'paired_vs_nn_full.csv',index=False)
    pd.DataFrame(query_deltas).to_csv(out/'online_query_deltas.csv',index=False)
    domination=[]
    for _,row in points[points.config.str.startswith('nn_')].iterrows():
        for label,mask in [('beam',points.config.eq('baseline')),('dabs',points.config.str.startswith('dabs_'))]:
            better=points[mask & points.recall.ge(row.recall) & points.ndc.le(row.ndc) &
                          (points.recall.gt(row.recall)|points.ndc.lt(row.ndc))].sort_values('ndc')
            domination.append(dict(config=row.config,ef=row.ef,control=label,dominated=not better.empty,
                dominator='' if better.empty else better.iloc[0].config,
                dominator_ef=np.nan if better.empty else better.iloc[0].ef,
                dominator_recall=np.nan if better.empty else better.iloc[0].recall,
                dominator_ndc=np.nan if better.empty else better.iloc[0].ndc))
    pd.DataFrame(domination).to_csv(out/'baseline_dominance.csv',index=False)
    target_rows=[]
    for target in (.92,.94,.95,.96,.97,.98,.99):
        for name,g in points.groupby('config'):
            # Means of decimal recall fractions can sit one ULP below an exact
            # target (e.g. .9699999999999998). This epsilon fixes representation
            # only; it is not the .005 experimental recall tolerance.
            best=g[g.recall.ge(target-1e-12)].sort_values('ndc').head(1)
            target_rows.append(dict(target_recall=target,config=name,reached=not best.empty,
                ef=np.nan if best.empty else best.iloc[0].ef,recall=np.nan if best.empty else best.iloc[0].recall,
                ndc=np.nan if best.empty else best.iloc[0].ndc))
    pd.DataFrame(target_rows).to_csv(out/'at_target_recall.csv',index=False)
    print(points[['config','ef','recall','ndc','ndc_mid','ndc_nn','gate_nn_cheap_scores','distance_equivalent_products']].to_string(index=False))
    print('NDC omits partial scores. Product-equivalent count omits sorting, query preparation and memory costs. Latency remains diagnostic.')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--indir',type=Path,required=True)
    a=p.parse_args(); m=json.loads((a.indir/'manifest.json').read_text(encoding='utf-8'))
    if m['inputs'] is None: raise ValueError('Dry-run plan contains no measured data')
    out=a.indir/'analysis'; out.mkdir(exist_ok=True)
    if m['suite']=='nn_diagnostic': diagnostic(a,m,out)
    elif m['suite']=='nn_online': online(a,m,out)
    else: raise ValueError('Wrong experiment suite')
if __name__=='__main__': main()
