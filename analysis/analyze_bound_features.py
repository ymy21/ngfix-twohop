#!/usr/bin/env python3
"""Describe jump/probe patterns using pre-probe state and local topology.

Requires pandas/numpy. Labels and final hop fractions are post-search diagnostics;
they are never predictor inputs. Query/ef pairs are kept separate in aggregates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['query_id','ef','cur_id','mid_id','hop']
STATE_FEATURES=['state_step','state_ndc','state_ndc_mid','state_ndc_nn','state_result_updates',
    'state_topk_updates','state_steps_since_topk_update','state_discovery_depth','state_frontier_size',
    'state_beam_size','state_topk_size','state_entry_dist','beam_min','beam_max','beam_mean',
    'beam_variance','beam_median','beam_q25','beam_q75','topk_min','topk_max','topk_mean',
    'topk_variance','topk_median','topk_q25','topk_q75']
PROBE_FEATURES=STATE_FEATURES+['mid_gap_normalized','cur_slack_normalized','topk_spread',
    'cur_total_degree','cur_repair_ratio','mid_total_degree','mid_repair_ratio',
    'cur_mid_data_dist','cur_mid_common_neighbors','cur_mid_jaccard']
JUMP_FEATURES=STATE_FEATURES+['mid_gap_normalized','nn_margin_normalized','topk_spread',
    'cur_total_degree','cur_repair_ratio','mid_total_degree','mid_repair_ratio','nn_total_degree',
    'nn_repair_ratio','cur_mid_data_dist','mid_nn_data_dist','cur_nn_data_dist',
    'cur_mid_common_neighbors','mid_nn_common_neighbors','cur_nn_common_neighbors',
    'cur_mid_jaccard','mid_nn_jaccard','cur_nn_jaccard','rank_in_fresh_nn_by_query_dist']


def joined(frame,states):
    if states.duplicated(KEY).any():raise ValueError('Duplicate pre-probe state keys')
    out=frame.merge(states,on=KEY,how='left',validate='many_to_one',indicator=True)
    if (out['_merge']!='both').any():raise ValueError('Events/probes missing matching pre-probe state')
    return out.drop(columns='_merge')


def add_features(df):
    df=df.copy()
    if 'bound_at_reject' in df:
        scale=np.maximum(np.abs(df.bound_at_reject),1e-6)
        df['mid_gap_normalized']=(df.mid_dist-df.bound_at_reject)/scale
        df['cur_slack_normalized']=(df.bound_at_reject-df.cur_dist)/scale
        df['probe_class']=np.where(df.admitted_nn==0,
            np.where(df.fresh_nn_cmps==0,'FAILED_NO_FRESH','FAILED_WITH_FRESH'),
            np.where((df.direct_new_gt+df.nearest_recovered_desc_gt)>0,'SUCCESS_RECOVERED_GT','SUCCESS_NO_RECOVERED_GT'))
        df['recovered_gt']=df.direct_new_gt+df.nearest_recovered_desc_gt
        df['useful']=(df.recovered_gt>0).astype(int)
    else:
        df['mid_gap_normalized']=(df.mid_dist-df.bound_at_mid_reject)/np.maximum(np.abs(df.bound_at_mid_reject),1e-6)
        df['nn_margin_normalized']=(df.bound_before_nn_admit-df.nn_dist)/np.maximum(np.abs(df.bound_before_nn_admit),1e-6)
    if 'topk_q75' in df:df['topk_spread']=df.topk_q75-df.topk_q25
    for role in ('cur','mid','nn'):
        column=role+'_repair_ratio'
        if column in df:
            df[role+'_region']=np.where(df[column]==0,'BASE_ONLY',np.where(df[column]==1,'REPAIR_ONLY','MIXED'))
    return df


def descriptions(df,group,features):
    rows=[]
    for keys,g in df.groupby(['ef',group],dropna=False):
        for feature in features:
            if feature not in g:continue
            x=pd.to_numeric(g[feature],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            if x.empty:continue
            rows.append(dict(ef=keys[0],group=keys[1],feature=feature,events=len(g),queries=g.query_id.nunique(),
                mean=x.mean(),median=x.median(),p25=x.quantile(.25),p75=x.quantile(.75),p90=x.quantile(.9)))
    return pd.DataFrame(rows)


def feature_bins(mids):
    rows=[]
    for ef,frame in mids.groupby('ef'):
        for feature in PROBE_FEATURES:
            if feature not in frame:continue
            values=pd.to_numeric(frame[feature],errors='coerce').replace([np.inf,-np.inf],np.nan)
            valid=frame.loc[values.notna()].copy();values=values.dropna()
            if values.empty:continue
            valid['feature_bin']=values.astype(str) if values.nunique()<=5 else pd.qcut(values,4,duplicates='drop').astype(str)
            for name,g in valid.groupby('feature_bin'):
                cost=g.fresh_nn_cmps.sum();recovered=g.recovered_gt.sum()
                rows.append(dict(ef=ef,feature=feature,feature_bin=name,probes=len(g),queries=g.query_id.nunique(),
                    success_rate=(g.admitted_nn>0).mean(),useful_rate=g.useful.mean(),fresh_nn_ndc=cost,
                    query_balanced_useful_rate=g.groupby('query_id').useful.mean().mean(),
                    recovered_gt=recovered,recovered_gt_per_1k_fresh_nn_ndc=1000*recovered/cost if cost else np.nan))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('mids','events','states','outdir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    states=pd.read_csv(a.states);mids=add_features(joined(pd.read_csv(a.mids),states))
    events=add_features(joined(pd.read_csv(a.events),states))
    descriptions(mids,'probe_class',PROBE_FEATURES).to_csv(a.outdir/'probe_feature_distributions.csv',index=False)
    descriptions(events,'jump_class',JUMP_FEATURES).to_csv(a.outdir/'jump_feature_distributions.csv',index=False)
    feature_bins(mids).to_csv(a.outdir/'probe_feature_yield_bins.csv',index=False)
    edge={0:'H',1:'N',2:'R'}
    events['edge_pair']=events.first_edge_type.map(edge)+events.second_edge_type.map(edge)
    events['local_region']=events.cur_region+'->'+events.mid_region+'->'+events.nn_region
    events['nn_visibility']=np.where(events.baseline_scored_nn==0,'UNSEEN',
        np.where(events.baseline_expanded_nn==0,'SCORED_NOT_EXPANDED','EXPANDED'))
    for column in ('edge_pair','local_region','nn_visibility'):
        summary=events.groupby(['ef','jump_class',column],dropna=False).agg(
            events=('nn_id','size'),queries=('query_id','nunique'),
            recovered_descendants=('nearest_recovered_gt_descendants','sum')).reset_index()
        summary.to_csv(a.outdir/('jump_by_'+column+'.csv'),index=False)
    reuse=[]
    for (ef,cls),g in events.groupby(['ef','jump_class']):
        for role in ('cur_id','mid_id','nn_id'):
            counts=g.groupby(role).size().sort_values(ascending=False)
            n=max(1,int(np.ceil(len(counts)*.01)))
            reuse.append(dict(ef=ef,jump_class=cls,role=role,unique_nodes=len(counts),
                top_1pct_nodes_event_share=counts.head(n).sum()/counts.sum() if len(counts) else np.nan))
    pd.DataFrame(reuse).to_csv(a.outdir/'node_reuse.csv',index=False)
    metadata=dict(probes=len(mids),jumps=len(events),queries_with_probe=int(mids.query_id.nunique()),
        snapshot_point='after rejected-mid scoring, before any second-hop scoring',
        online_observables=PROBE_FEATURES,labels=['probe_class','useful','recovered_gt','jump_class'],
        notes=['local_region is repair-edge composition, not community detection',
               'discovery_depth is path depth in the discovery tree, not shortest graph distance',
               'edge geometry and neighbor overlap are offline topology features with nonzero extraction cost',
               'feature bins describe correlations; validate rules on independent query IDs',
               'DARTH-inspired features; this does not reproduce or evaluate DARTH'])
    (a.outdir/'feature_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Feature summaries:',a.outdir)


if __name__=='__main__':main()
