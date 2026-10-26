#!/usr/bin/env python3
"""Read-only, exhaustive consolidation of the two graph/K experiment suites.

Deep statistics use column-major temporary memmaps to compute exact quantiles
without loading the 8.9 GiB trace set at once. Source files are never changed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PERF=ROOT/'result/graph_k_original_ngfixgrid_1k_v1'
MECH=ROOT/'result/bound_features_selected_1k_v1'
OUT=ROOT/'result/complete_experiment_report_20261013'
TABLES=OUT/'tables'
SCRATCH=OUT/'_scratch'
CORE=['trace.csv','trace_jump_events.csv','trace_jump_mids.csv','trace_probe_state.csv','trace_jump_hopbins.csv']
QUANTILES=[.01,.05,.25,.5,.75,.9,.95,.99]
QUANTILE_NAMES=['p01','p05','p25','median','p75','p90','p95','p99']
KEY=['query_id','ef','cur_id','mid_id','hop']
CLASS_MID=['FAILED_NO_FRESH','FAILED_WITH_FRESH','SUCCESS_NO_RECOVERED_GT','SUCCESS_RECOVERED_GT']
CLASS_EVENT=['DIRECT_GT_NEW','DIRECT_GT_SHARED','ROUTER_NO_GT','ROUTER_TO_GT','SANITY_GT_ADMITTED_NOT_FINAL']


def save_json(path,value):
    def normalize(x):
        if isinstance(x,dict):return {str(k):normalize(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [normalize(v) for v in x]
        if isinstance(x,(float,np.floating)):return float(x) if math.isfinite(x) else None
        if isinstance(x,np.integer):return int(x)
        if isinstance(x,np.bool_):return bool(x)
        if isinstance(x,Path):return str(x)
        return x
    def convert(x):
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.floating,)):return float(x) if math.isfinite(x) else None
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(normalize(value),ensure_ascii=False,indent=2,default=convert,allow_nan=False)+'\n',encoding='utf-8')


def save_table(name,frame):
    if not isinstance(frame,pd.DataFrame):frame=pd.DataFrame(frame)
    frame.to_csv(TABLES/(name+'.csv'),index=False,encoding='utf-8-sig')
    return frame


def case_info(folder):
    graph,k=re.fullmatch(r'(hnsw|ngfix)_k(10|100)',folder.parent.name).groups()
    return dict(case=f'{graph}_k{k}_ef{folder.name[2:]}',graph=graph,k=int(k),ef=int(folder.name[2:]))


def source_line(file,needle):
    path=ROOT/file
    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if needle in line:return f'{file}:{i}'
    return file


def setup():
    OUT.mkdir(exist_ok=True);TABLES.mkdir(exist_ok=True);SCRATCH.mkdir(exist_ok=True)


def inventory():
    records=[]
    for suite,root in [('performance',PERF),('mechanism',MECH)]:
        for path in sorted(root.rglob('*')):
            if not path.is_file():continue
            row=dict(suite=suite,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,
                     mtime_ns=path.stat().st_mtime_ns,extension=path.suffix,rows=None,columns=None)
            if path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as f:
                    reader=csv.reader(f);header=next(reader,[])
                    row['columns']=len(header);row['header']=' | '.join(header)
                    if path.stat().st_size<10_000_000:row['rows']=sum(1 for _ in reader)
            records.append(row)
    save_table('source_inventory',records)
    return records


def small_analysis():
    from analyze_hnsw_stage1 import summarize,match_targets
    from compare_graph_k_matrix import check_pair
    checks=[];performance=[];queries=[];latencies=[];manifests={}
    for graph in ['hnsw','ngfix']:
        for k in [10,100]:
            d=PERF/f'{graph}_k{k}'/'clean'
            m=json.loads((d/'manifest.json').read_text(encoding='utf-8'));manifests[graph,k]=m
            t=pd.read_csv(d/'timings.csv');q=pd.read_csv(d/'timings.csv.queries.csv');l=pd.read_csv(d/'timings.csv.latencies.csv')
            points=summarize(t.to_dict('records'),q.to_dict('records'),m)
            for p in points:p.update(graph=graph,k=k)
            for frame in [q,l]:frame['graph']=graph;frame['k']=k
            performance.extend(points);queries.append(q);latencies.append(l)
            checks.append(dict(check='completed performance',case=f'{graph}_k{k}',passed=m['completed'],details=f'{len(t)} points, {len(q)} query rows'))
            grouped=l.groupby(['method','ef'])
            for _,r in t.iterrows():
                g=grouped.get_group((r['method'],r['ef']))
                ok=len(g)==1000 and abs(g.latency_us.sum()/1e6-r.seconds)<1e-9 and abs(g.ndc.mean()-r.mean_ndc)<1e-9
                checks.append(dict(check='latency/QPS/NDC reconciliation',case=f'{graph}_k{k}_ef{r.ef}_{r.method}',passed=ok,details='sum per-query search time vs timed seconds'))
    for k in [10,100]:check_pair(manifests['hnsw',k],manifests['ngfix',k])
    p=pd.DataFrame(performance);q=pd.concat(queries,ignore_index=True);l=pd.concat(latencies,ignore_index=True)
    latency_stats=l.groupby(['graph','k','method','ef']).latency_us.agg(
        latency_mean_us='mean',latency_p50_us='median',latency_p95_us=lambda x:x.quantile(.95),
        latency_p99_us=lambda x:x.quantile(.99),latency_max_us='max').reset_index()
    p=p.merge(latency_stats,on=['graph','k','method','ef'],validate='one_to_one')
    save_table('performance_all_points',p);save_table('performance_queries',q);save_table('performance_latencies',l)
    matched=[];targets=[]
    for (graph,k),g in p.groupby(['graph','k']):
        for t in g[g.method=='bound_unbounded'].to_dict('records'):
            b=g[(g.method=='beam')&(g.recall>=t['recall']-1e-12)]
            best=b.loc[b.qps_median.idxmax()] if len(b) else None
            matched.append(dict(graph=graph,k=k,bound_ef=t['ef'],bound_recall=t['recall'],bound_qps=t['qps_median'],bound_ndc=t['mean_ndc'],
                beam_reached=best is not None,beam_ef=None if best is None else best.ef,beam_recall=None if best is None else best.recall,
                beam_qps=None if best is None else best.qps_median,beam_ndc=None if best is None else best.mean_ndc,
                beam_speedup=None if best is None else best.qps_median/t['qps_median'],
                bound_ndc_ratio=None if best is None else t['mean_ndc']/best.mean_ndc))
        for r in match_targets(g.to_dict('records'),[.8,.85,.9,.94,.95,.97,.98,.99,.995,.999]):
            targets.append(dict(graph=graph,k=k,**r))
    save_table('matched_recall_points',matched);save_table('target_recall',targets)
    paired=[];query_dist=[];existing=[];eh=[];edges=[];bins=[];feature_trends=[]
    qlookup=q.set_index(['graph','k','method','ef','query_id'])
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder)
        raw=pd.read_csv(folder/'trace.csv');raw=raw[raw.row_type=='QUERY'].copy()
        if len(raw)!=1000 or raw.query_id.nunique()!=1000:raise ValueError('Incomplete case '+info['case'])
        for method,rcol,ncol in [('beam','paired_baseline_recall','paired_baseline_ndc'),('bound_unbounded','recall','ndc')]:
            ids=[(info['graph'],info['k'],method,info['ef'],int(v)) for v in raw.query_id]
            ref=qlookup.loc[ids]
            ok=np.allclose(raw[rcol].to_numpy(),ref.recall.to_numpy(),atol=1e-6,rtol=0) and np.array_equal(raw[ncol].to_numpy(),ref.ndc.to_numpy())
            checks.append(dict(check='diagnostic vs performance',case=info['case']+'_'+method,passed=bool(ok),details='all 1000 query IDs, recall and NDC'))
        for title,ok in [
            ('NDC partition',np.array_equal(raw.ndc,raw.ndc_entry+raw.ndc_mid+raw.ndc_nn)),
            ('NN cost partition',np.array_equal(raw.ndc_nn,raw.ndc_nn_jump+raw.ndc_nn_nojump)),
            ('probe outcome partition',np.array_equal(raw.twohop_mid_scans,raw.jump_mids+raw.failed_jump_mids)),
            ('GT/recall reconciliation',np.allclose(raw.recall-raw.paired_baseline_recall,(raw.recovered_gt_total-raw.lost_gt_total)/info['k'],atol=1e-6)),
            ('direct/routing GT partition',np.array_equal(raw.recovered_gt_total,raw.recovered_gt_direct_jump+raw.recovered_gt_bridge_descendant)),
            ('distance edge partition',np.array_equal(raw.scored_total+1,raw.ndc))]:
            checks.append(dict(check=title,case=info['case'],passed=bool(ok),details='per-query exact counts / rounded recall tolerance'))
        s=pd.read_csv(folder/'summary/paired_mechanism_summary.csv').iloc[0].to_dict()
        s.update(info);s['ndc_ratio']=s['ndc_mean']/s['paired_baseline_ndc_mean']
        s['ndc_nn_share']=s['ndc_nn_mean']/s['ndc_mean'];paired.append(s)
        for f in ['recall','paired_baseline_recall','ndc','paired_baseline_ndc','recovered_gt_total','lost_gt_total','hops','twohop_mid_scans']:
            x=raw[f];query_dist.append(dict(**info,field=f,count=len(x),mean=x.mean(),std=x.std(ddof=0),min=x.min(),p05=x.quantile(.05),p25=x.quantile(.25),median=x.median(),p75=x.quantile(.75),p95=x.quantile(.95),p99=x.quantile(.99),max=x.max()))
        for path in sorted(folder.rglob('*.csv')):
            if path.name in CORE or path.stat().st_size>10_000_000:continue
            frame=pd.read_csv(path)
            for _,r in frame.iterrows():existing.append(dict(**info,table=str(path.relative_to(folder)),values=r.to_dict()))
        ep=pd.read_csv(folder/'summary/twohop_edge_pair_summary.csv');ep['case']=info['case'];ep['graph']=info['graph'];ep['k']=info['k'];edges.append(ep)
        fb=pd.read_csv(folder/'features/probe_feature_yield_bins.csv');fb['case']=info['case'];fb['graph']=info['graph'];fb['k']=info['k'];bins.append(fb)
        for feature,g in fb.groupby('feature'):
            def lower(text):
                m=re.search(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',str(text));return float(m.group()) if m else 0
            g=g.assign(order=g.feature_bin.map(lower)).sort_values('order')
            lo,hi=g.iloc[0],g.iloc[-1]
            feature_trends.append(dict(**info,feature=feature,bins=len(g),low_bin=lo.feature_bin,high_bin=hi.feature_bin,
                low_useful=lo.useful_rate,high_useful=hi.useful_rate,low_query_balanced=lo.query_balanced_useful_rate,
                high_query_balanced=hi.query_balanced_useful_rate,low_roi=lo.recovered_gt_per_1k_fresh_nn_ndc,
                high_roi=hi.recovered_gt_per_1k_fresh_nn_ndc,
                useful_rate_range=g.useful_rate.max()-g.useful_rate.min(),
                pattern='constant / one bin' if len(g)==1 else 'higher values more useful' if hi.useful_rate>lo.useful_rate else 'lower values more useful' if hi.useful_rate<lo.useful_rate else 'no end-bin difference'))
        hs=pd.read_csv(folder/'trace_jump_hopbins.csv')
        for name,g in hs.groupby('bin'):
            eh.append(dict(**info,bin=int(name),probes=g.scans.sum(),successful=g.successful_mids.sum(),useful=g.useful_mids.sum(),
                nn_ndc=g.fresh_nn_ndc.sum(),direct_new=g.direct_new_gt.sum(),recovered_desc=g.recovered_desc_gt.sum()))
    save_table('mechanism_cases',paired);save_table('mechanism_query_distributions',query_dist)
    save_table('edge_pairs',pd.concat(edges,ignore_index=True));save_table('probe_feature_bins',pd.concat(bins,ignore_index=True));save_table('feature_trends',feature_trends)
    save_table('search_phase_bins',eh);save_table('validation_checks',checks)
    save_json(OUT/'existing_tables_all.json',existing)
    save_json(OUT/'analysis_context.json',dict(performance_points=len(p),cases=len(paired),manifests={f'{g}_k{k}':m for (g,k),m in manifests.items()},
        source_files=inventory(),stage='small complete',created_date='2026-10-13'))
    print('Small tables consolidated:',len(p),'performance points;',len(paired),'mechanism cases; checks',Counter(x['passed'] for x in checks),flush=True)


def stats_of(values):
    x=np.asarray(values);valid=np.isfinite(x);v=x[valid];n=len(v)
    out=dict(rows=len(x),valid=n,missing=len(x)-n)
    if not n:return dict(out,zero_count=0,unique=0,sum=None,mean=None,std=None,min=None,max=None,**{k:None for k in QUANTILE_NAMES})
    lo=float(v.min());hi=float(v.max());constant=lo==hi
    quantiles=[lo]*len(QUANTILES) if constant else np.quantile(v,QUANTILES,method='linear').tolist()
    return dict(out,zero_count=int(np.count_nonzero(v==0)),unique=1 if constant else int(len(np.unique(v))),
        sum=float(v.sum()),mean=float(v.mean()),std=float(v.std(ddof=0)),min=lo,max=hi,**dict(zip(QUANTILE_NAMES,quantiles)))


def progress(message):
    print(message,flush=True)
    save_json(OUT/'progress.json',dict(stage=message,time=time.time()))


def deep_file(folder,name):
    info=case_info(folder);path=folder/name;target=OUT/'deep_cache'/f"{info['case']}_{path.stem}.json"
    if target.exists():return json.loads(target.read_text(encoding='utf-8'))
    meta=json.loads((folder/'features/feature_manifest.json').read_text(encoding='utf-8'))
    expected=meta['jumps'] if name=='trace_jump_events.csv' else meta['probes'] if name in ['trace_jump_mids.csv','trace_probe_state.csv'] else None
    sample=pd.read_csv(path,nrows=2000)
    if name=='trace.csv':sample=sample[sample.row_type=='QUERY']
    numeric=[c for c in sample if pd.api.types.is_numeric_dtype(sample[c])]
    categorical=[c for c in sample if c not in numeric]
    if expected is None:
        count=sum(1 for _ in path.open(encoding='utf-8'))-1
        expected=count-1 if name=='trace.csv' else count
    mmap_path=SCRATCH/f"{info['case']}_{path.stem}.bin"
    matrix=np.memmap(mmap_path,dtype='float64',mode='w+',shape=(len(numeric),expected))
    class_path=SCRATCH/f"{info['case']}_{path.stem}_classes.bin"
    classes=np.memmap(class_path,dtype='int8',mode='w+',shape=(expected,));classes[:]=-1
    class_names=CLASS_MID if name in ['trace_jump_mids.csv','trace_probe_state.csv'] else CLASS_EVENT if name=='trace_jump_events.csv' else []
    inherited=None
    if name=='trace_probe_state.csv':
        inherited=np.memmap(SCRATCH/f"{info['case']}_mid_labels.bin",dtype='int8',mode='r',shape=(expected,))
    cats={c:Counter() for c in categorical};offset=0;hash_keys=hashlib.sha256();sum_checks=Counter();outcomes={};id_counts={}
    for c in ['cur_id','mid_id','nn_id']:
        if c in numeric:id_counts[c]=np.zeros(2505000,dtype='int64')
    for chunk in pd.read_csv(path,chunksize=40000):
        if name=='trace.csv':chunk=chunk[chunk.row_type=='QUERY']
        n=len(chunk)
        if not n:continue
        if offset+n>expected:raise ValueError('More rows than manifest: '+str(path))
        if all(c in chunk for c in KEY):hash_keys.update(np.ascontiguousarray(chunk[KEY].to_numpy(dtype='<i8')).tobytes())
        if name=='trace_jump_mids.csv':
            useful=(chunk.direct_new_gt+chunk.nearest_recovered_desc_gt)>0
            cl=np.where(chunk.admitted_nn==0,np.where(chunk.fresh_nn_cmps==0,0,1),np.where(useful,3,2))
            for i,label in enumerate(CLASS_MID):
                g=chunk[cl==i];o=outcomes.setdefault(label,dict(probes=0,nn_ndc=0,scans=0,admitted=0,recovered_gt=0))
                o['probes']+=len(g);o['nn_ndc']+=int(g.fresh_nn_cmps.sum());o['scans']+=int(g.checked_nn.sum());o['admitted']+=int(g.admitted_nn.sum());o['recovered_gt']+=int((g.direct_new_gt+g.nearest_recovered_desc_gt).sum())
        elif name=='trace_probe_state.csv':
            cl=inherited[offset:offset+n]
            sum_checks['bad_ndc_partition']+=int(np.count_nonzero(chunk.state_ndc!=1+chunk.state_ndc_mid+chunk.state_ndc_nn))
            sum_checks['bad_step']+=int(np.count_nonzero(chunk.state_step!=chunk.hop))
            sum_checks['bad_topk_size']+=int(np.count_nonzero(chunk.state_topk_size!=np.minimum(info['k'],chunk.state_beam_size)))
            sum_checks['bad_distance_order']+=int(np.count_nonzero((chunk.topk_min>chunk.topk_max+1e-9)|(chunk.topk_max>chunk.beam_max+1e-9)))
        elif name=='trace_jump_events.csv':
            cl=chunk.jump_class.map({s:i for i,s in enumerate(CLASS_EVENT)}).to_numpy()
            if pd.isna(cl).any():raise ValueError('Unknown jump class')
            sum_checks['scored_and_unexpanded']+=int(((chunk.baseline_scored_nn==1)&(chunk.baseline_expanded_nn==0)).sum())
            sum_checks['unseen']+=int((chunk.baseline_scored_nn==0).sum())
            sum_checks['expanded']+=int((chunk.baseline_expanded_nn==1).sum())
            sum_checks['unique_recovered_attribution']+=int((chunk.jump_class=='DIRECT_GT_NEW').sum()+chunk.nearest_recovered_gt_descendants.sum())
        else:cl=np.full(n,-1,dtype='int8')
        classes[offset:offset+n]=cl
        # Mask placeholder best-neighbor structure when the probe found NO fresh NN.
        absent=chunk.best_scored_nn<0 if name=='trace_jump_mids.csv' else None
        for i,c in enumerate(numeric):
            values=chunk[c].to_numpy(dtype='float64',copy=True)
            if absent is not None and (c.startswith('best_nn_') or c.startswith('mid_best_nn_') or c.startswith('cur_best_nn_') or c.startswith('best_scored_')):
                values[absent.to_numpy()]=np.nan
            matrix[i,offset:offset+n]=values
            if c in id_counts:
                ids=chunk[c].to_numpy(dtype='int64');id_counts[c]+=np.bincount(ids[ids>=0],minlength=2505000)
        for c in categorical:cats[c].update(chunk[c].fillna('<missing>').astype(str).value_counts().to_dict())
        offset+=n
    if offset!=expected:raise ValueError('Row count mismatch: '+str(path))
    matrix.flush();classes.flush();field_stats=[];class_stats=[];categorical_stats=[]
    for i,c in enumerate(numeric):
        field_stats.append(dict(field=c,**stats_of(matrix[i])))
        if class_names and c not in ['query_id','ef','cur_id','mid_id','nn_id','probe_id']:
            for cls,label in enumerate(class_names):
                mask=classes==cls
                if mask.any():class_stats.append(dict(field=c,group=label,**stats_of(matrix[i,mask])))
        # Preserve exact value counts for boolean/sentinel/edge-type/step/bin fields
        # with small cardinality; high-cardinality identifiers get node reuse summaries.
        valid=matrix[i][np.isfinite(matrix[i])]
        if len(valid) and field_stats[-1]['unique']<=256:
            vals,counts=np.unique(valid,return_counts=True)
            categorical_stats.extend(dict(field=c,value=float(v),count=int(n),share=float(n)/len(valid)) for v,n in zip(vals,counts))
    for c,values in cats.items():
        categorical_stats.extend(dict(field=c,value=v,count=n,share=n/offset) for v,n in values.items())
    nodes=[]
    for c,counts in id_counts.items():
        active=counts[counts>0];ordered=np.sort(active)[::-1];total=int(ordered.sum())
        nodes.append(dict(role=c,unique_nodes=len(active),events=total,max_reuse=int(ordered[0]) if len(ordered) else 0,
            top1pct_share=float(ordered[:max(1,math.ceil(len(ordered)*.01))].sum())/total if total else None,
            top5pct_share=float(ordered[:max(1,math.ceil(len(ordered)*.05))].sum())/total if total else None))
    result=dict(**info,file=name,rows=offset,fields=numeric+categorical,field_stats=field_stats,class_stats=class_stats,
        categorical_stats=categorical_stats,outcomes=outcomes,nodes=nodes,checks=dict(sum_checks),key_sha256=hash_keys.hexdigest())
    (OUT/'deep_cache').mkdir(exist_ok=True);save_json(target,result)
    if name=='trace_jump_mids.csv':
        keep=np.memmap(SCRATCH/f"{info['case']}_mid_labels.bin",dtype='int8',mode='w+',shape=(expected,));keep[:]=classes[:];keep.flush();del keep
    del matrix,classes
    if inherited is not None:del inherited
    # Only task-created temporary files inside the verified scratch directory.
    for temporary in [mmap_path,class_path]:
        if temporary.resolve().parent!=SCRATCH.resolve():raise ValueError('Unexpected scratch path')
        temporary.unlink()
    return result


def deep_analysis():
    records=[]
    for folder in sorted(MECH.glob('*_k*/ef*')):
        for name in ['trace.csv','trace_jump_events.csv','trace_jump_mids.csv','trace_probe_state.csv','trace_jump_hopbins.csv']:
            progress('Processing '+case_info(folder)['case']+' '+name)
            records.append(deep_file(folder,name))
        keep=SCRATCH/f"{case_info(folder)['case']}_mid_labels.bin"
        if keep.exists():keep.unlink()
    stats=[];class_stats=[];cats=[];outcomes=[];nodes=[];checks=[]
    for r in records:
        info={k:r[k] for k in ['case','graph','k','ef','file']}
        stats.extend(dict(**info,**s) for s in r['field_stats']);class_stats.extend(dict(**info,**s) for s in r['class_stats'])
        cats.extend(dict(**info,**s) for s in r['categorical_stats']);nodes.extend(dict(**info,**s) for s in r['nodes'])
        outcomes.extend(dict(**info,group=label,**s) for label,s in r['outcomes'].items())
        for name,value in r['checks'].items():checks.append(dict(**info,check=name,value=value))
    for name,data in [('all_field_distributions',stats),('all_class_field_distributions',class_stats),
                      ('all_categorical_distributions',cats),('probe_outcomes',outcomes),('node_reuse_all',nodes),('deep_checks',checks)]:save_table(name,data)
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder);subset=[r for r in records if r['case']==info['case']]
        byname={r['file']:r for r in subset}
        if byname['trace_jump_mids.csv']['key_sha256']!=byname['trace_probe_state.csv']['key_sha256']:raise ValueError('Probe/state row keys mismatch')
    save_json(OUT/'deep_manifest.json',dict(files=[{k:r[k] for k in ['case','file','rows','key_sha256']} for r in records],
        quantiles='exact, numpy linear interpolation; not sampled',placeholder_policy='best-scored-neighbor placeholder columns masked when best_scored_nn<0'))
    progress('Deep statistics complete')


def post_analysis():
    from report_field_catalog import definition
    records=[];unmapped=[];schemas=set();metrics=[]
    for root in [PERF,MECH]:
        for path in sorted(root.rglob('*.csv')):
            frame=pd.read_csv(path,nrows=0)
            scope=path.name
            for field in frame.columns:
                key=(scope,field)
                if key in schemas:continue
                schemas.add(key);d=definition(field,scope)
                if d is None:unmapped.append(key);continue
                records.append(dict(scope=scope,field=field,**d))
    if unmapped:raise ValueError('Undefined source fields '+str(unmapped))
    derived=['mid_gap_normalized','cur_slack_normalized','nn_margin_normalized','topk_spread','probe_class','useful','recovered_gt']
    for field in derived:records.append(dict(scope='derived_features',field=field,**definition(field)))
    save_table('field_dictionary',records)
    inv=pd.read_csv(TABLES/'source_inventory.csv')
    deep=json.loads((OUT/'deep_manifest.json').read_text(encoding='utf-8'))
    for r in deep['files']:
        graph,k,ef=re.fullmatch(r'(hnsw)_k(10|100)_ef(\d+)',r['case']).groups() if r['case'].startswith('hnsw_') else re.fullmatch(r'(ngfix)_k(10|100)_ef(\d+)',r['case']).groups()
        suffix=str(Path('result/bound_features_selected_1k_v1')/f'{graph}_k{k}'/f'ef{ef}'/r['file'])
        inv.loc[inv.path==suffix,'rows']=r['rows']+(1 if r['file']=='trace.csv' else 0)
    save_table('source_inventory',inv)
    checks=pd.read_csv(TABLES/'validation_checks.csv').to_dict('records')
    deep_checks=pd.read_csv(TABLES/'deep_checks.csv')
    for _,r in deep_checks[deep_checks.check.str.startswith('bad')].iterrows():
        checks.append(dict(check=r.check,case=r['case'],passed=r.value==0,details=f'{r.file}, bad rows={int(r.value)}'))
    caches={}
    for p in (OUT/'deep_cache').glob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8'));caches[r['case'],r['file']]=r
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder);by={f:caches[info['case'],f] for f in CORE}
        summ=pd.read_csv(folder/'summary/paired_mechanism_summary.csv').iloc[0]
        def total(file,field):return next(s['sum'] for s in by[file]['field_stats'] if s['field']==field)
        tests=[('probe/state key alignment',by['trace_jump_mids.csv']['key_sha256']==by['trace_probe_state.csv']['key_sha256']),
            ('all probe NDC reconciliation',total('trace_jump_mids.csv','fresh_nn_cmps')==total('trace.csv','ndc_nn')),
            ('all admitted jump reconciliation',by['trace_jump_events.csv']['rows']==total('trace.csv','jump_candidates')),
            ('hop bins NN cost reconciliation',total('trace_jump_hopbins.csv','fresh_nn_ndc')==total('trace.csv','ndc_nn')),
            ('unique GT event attribution',by['trace_jump_events.csv']['checks']['unique_recovered_attribution']==total('trace.csv','recovered_gt_total'))]
        for name,ok in tests:checks.append(dict(check=name,case=info['case'],passed=ok,details='independent full-row accumulation'))
    save_table('validation_checks',checks)
    # Same graph/ef gives the same physical trace for K=10 and K=100;
    # GT labels and state top-K summaries differ, but core search does not.
    same=[]
    for graph in ['hnsw','ngfix']:
        a=caches[f'{graph}_k10_ef100','trace_jump_mids.csv'];b=caches[f'{graph}_k100_ef100','trace_jump_mids.csv']
        same.append(dict(graph=graph,ef=100,probes_k10=a['rows'],probes_k100=b['rows'],same_probe_path_keys=a['key_sha256']==b['key_sha256']))
    save_table('same_ef_k_trace_checks',same)
    sparse=[];ancestry=[];locations=[]
    columns=['cur_mid_jaccard','fresh_nn_cmps','admitted_nn','checked_nn','direct_new_gt','nearest_recovered_desc_gt',
        'ancestor_recovered_desc_gt','baseline_scored_cur','baseline_expanded_cur','baseline_scored_mid','baseline_expanded_mid']
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder);sp={};an={};loc={}
        for chunk in pd.read_csv(folder/'trace_jump_mids.csv',usecols=columns,chunksize=60000):
            recovered=chunk.direct_new_gt+chunk.nearest_recovered_desc_gt
            near=recovered>0;ancestor=(chunk.direct_new_gt+chunk.ancestor_recovered_desc_gt)>0
            location=np.where(chunk.baseline_expanded_mid==1,'BASELINE_MID_EXPANDED',
                np.where((chunk.baseline_expanded_cur==1)&(chunk.baseline_scored_mid==1),'SHARED_CUR_MID_NOT_EXPANDED','NEW_UPSTREAM_CUR'))
            selections=[('jaccard',np.where(chunk.cur_mid_jaccard==0,'ZERO','POSITIVE'),sp),
                        ('attribution',np.where(near,'NEAREST_USEFUL',np.where(ancestor,'ANCESTOR_ONLY_USEFUL','NO_RECOVERED_ANCESTRY')),an),
                        ('baseline_path',location,loc)]
            for family,labels,acc in selections:
                for label in np.unique(labels):
                    mask=labels==label;g=chunk.loc[mask];r=recovered.loc[mask]
                    z=acc.setdefault(label,dict(probes=0,nn_ndc=0,successful=0,nearest_useful=0,ancestor_useful=0,recovered_gt=0))
                    z['probes']+=len(g);z['nn_ndc']+=int(g.fresh_nn_cmps.sum());z['successful']+=int((g.admitted_nn>0).sum())
                    z['nearest_useful']+=int(near.loc[mask].sum());z['ancestor_useful']+=int(ancestor.loc[mask].sum());z['recovered_gt']+=int(r.sum())
        for acc,output in [(sp,sparse),(an,ancestry),(loc,locations)]:
            for label,v in acc.items():
                output.append(dict(**info,group=label,**v,useful_rate=v['nearest_useful']/v['probes'],
                    ancestor_useful_rate=v['ancestor_useful']/v['probes'],
                    recovered_gt_per_1k_nn_ndc=1000*v['recovered_gt']/v['nn_ndc'] if v['nn_ndc'] else None))
        for _,r in pd.read_csv(folder/'trace_repair_degree_summary.csv').iterrows():metrics.append(dict(**info,metric=r.metric,value=r.value))
    save_table('sparse_jaccard_controls',sparse);save_table('ancestry_cost_groups',ancestry);save_table('baseline_path_groups',locations);save_table('graph_metrics',metrics)
    # Aggregate event edge composition with a unique recovered-GT numerator.
    pair=pd.read_csv(TABLES/'edge_pairs.csv');recovery=[];visibility=[];regions=[]
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder)
        e=pd.read_csv(folder/'features/jump_by_edge_pair.csv')
        for edge,g in e.groupby('edge_pair'):
            recovery.append(dict(case=info['case'],edge_pair=edge,direct_new_gt=int(g.loc[g.jump_class=='DIRECT_GT_NEW','events'].sum()),
                recovered_descendants=int(g.recovered_descendants.sum())))
        for file,output in [('jump_by_nn_visibility.csv',visibility),('jump_by_local_region.csv',regions)]:
            f=pd.read_csv(folder/'features'/file)
            output.extend(dict(info,**r) for r in f.to_dict('records'))
    pair=pair.merge(pd.DataFrame(recovery),on=['case','edge_pair'],how='left').fillna({'direct_new_gt':0,'recovered_descendants':0})
    pair['attributed_recovered_gt']=pair.direct_new_gt+pair.recovered_descendants
    pair['recovered_gt_per_1k_scores']=np.where(pair.fresh_scored>0,1000*pair.attributed_recovered_gt/pair.fresh_scored,np.nan)
    for quantity in ['examined','fresh_scored','admitted','attributed_recovered_gt']:
        pair[quantity+'_share']=pair[quantity]/pair.groupby('case')[quantity].transform('sum')
    save_table('edge_pair_recovery',pair);save_table('jump_visibility',visibility);save_table('jump_regions',regions)
    # Query-subgroup throughput is reconstructed from measured query times,
    # not independently benchmarked. The IDs are fixed from HNSW beam ef=200.
    q=pd.read_csv(TABLES/'performance_queries.csv');l=pd.read_csv(TABLES/'performance_latencies.csv')
    merged=q.merge(l,on=['graph','k','method','ef','query_id'],suffixes=('','_timed'),validate='one_to_one')
    hard=[]
    for k in [10,100]:
        anchor=q[(q.k==k)&(q.graph=='hnsw')&(q.method=='beam')&(q.ef==200)].sort_values(['recall','ndc','query_id'],ascending=[True,False,True])
        ids=set(anchor.head(200).query_id)
        for keys,g in merged[merged.k==k].groupby(['graph','method','ef']):
            for label,mask in [('all',np.ones(len(g),dtype=bool)),('fixed_low_recall_20pct',g.query_id.isin(ids)),('remaining_80pct',~g.query_id.isin(ids))]:
                s=g.loc[mask]
                hard.append(dict(graph=keys[0],k=k,method=keys[1],ef=keys[2],group=label,queries=len(s),recall=s.recall.mean(),
                    mean_ndc=s.ndc.mean(),latency_mean_us=s.latency_us.mean(),reconstructed_qps=len(s)*1e6/s.latency_us.sum()))
    save_table('fixed_query_group_performance',hard)
    progress('Post-analysis complete; all source columns defined')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['small','deep','post'],required=True);a=p.parse_args();setup()
    if a.phase=='small':small_analysis()
    elif a.phase=='deep':deep_analysis()
    else:post_analysis()


if __name__=='__main__':main()
