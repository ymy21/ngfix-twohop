#!/usr/bin/env python3
"""Validate clean timing/query data and summarize capped-prefix ablations."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile(values,fraction):
    values=sorted(values);x=(len(values)-1)*fraction;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def attach_latencies(points,latencies,queries,p):
    expected={(r['method'],r['ef']) for r in points};groups={};seen=set()
    ndcs={(r['method'],int(r['ef']),int(r['query_id'])):int(r['ndc']) for r in queries}
    for r in latencies:
        key=r['method'],int(r['ef']);rnd=int(r['round']);qid=int(r['query_id']);v=float(r['latency_us'])
        if key not in expected or (*key,rnd,qid) in seen or not 0<=rnd<p['rounds'] or not p['offset']<=qid<p['offset']+p['limit']:
            raise ValueError('Unexpected/duplicate latency')
        if not math.isfinite(v) or v<0 or int(r['ndc'])!=ndcs[(*key,qid)]:raise ValueError('Invalid latency/NDC')
        seen.add((*key,rnd,qid));groups.setdefault(key,[]).append(v)
    if set(groups)!=expected or any(len(v)!=p['limit']*p['rounds'] for v in groups.values()):raise ValueError('Missing latencies')
    for r in points:
        values=groups[r['method'],r['ef']]
        r['latency_mean_us']=statistics.mean(values)
        for name,f in [('p50',.5),('p95',.95),('p99',.99)]:r['latency_'+name+'_us']=quantile(values,f)


def summarize_case(case,timings,queries,p):
    expected={('beam',ef) for ef in case['beam_efs']}|{(case['other_method'],ef) for ef in case['other_efs']}
    groups={};qgroups={};seen=set();qseen=set()
    for r in timings:
        key=r['method'],int(r['ef']);rnd=int(r['round'])
        if key not in expected or (*key,rnd) in seen or not 0<=rnd<p['rounds']:raise ValueError('Unexpected/duplicate timing')
        seen.add((*key,rnd))
        if (int(r['k']),int(r['queries']),int(r['query_offset']),int(r['verified_queries']))!=(p['k'],p['limit'],p['offset'],min(p['verify_limit'],p['limit'])):
            raise ValueError('Timing population mismatch')
        qps,seconds=float(r['qps']),float(r['seconds']);batch=int(r['batch_queries'])
        if min(qps,seconds)<=0 or not all(math.isfinite(x) for x in (qps,seconds)) or not math.isclose(qps,batch/seconds,rel_tol=1e-10):
            raise ValueError('Invalid QPS accounting')
        if r['timing_protocol']!=p['timing_protocol'] or batch<p['limit'] or batch%p['limit']:
            raise ValueError('Timing protocol mismatch')
        if (p['timing_protocol']=='original' and batch!=p['limit']) or (p['timing_protocol']=='batch' and seconds*1000+1e-6<p['min_batch_ms']):
            raise ValueError('Invalid timing duration')
        groups.setdefault(key,[]).append(r)
    for r in queries:
        key=r['method'],int(r['ef']);qid=int(r['query_id']);recall=float(r['recall']);ndc=int(r['ndc'])
        if key not in expected or (*key,qid) in qseen or not p['offset']<=qid<p['offset']+p['limit']:raise ValueError('Unexpected/duplicate query')
        if not math.isfinite(recall) or not 0<=recall<=1 or ndc<1:raise ValueError('Invalid query result')
        qseen.add((*key,qid));qgroups.setdefault(key,[]).append(r)
        if key[0]=='selective_prefix':
            c={name:int(r[name]) for name in ('ndc_entry','ndc_mid','ndc_nn','rejected_mids','probes','blocked_cur','blocked_gap','blocked_budget','blocked_selection','successful_probes','admitted_nn','scanned_nn','expanded')}
            if min(c.values())<0 or c['ndc_entry']!=1 or ndc!=c['ndc_entry']+c['ndc_mid']+c['ndc_nn']:
                raise ValueError('Distance accounting mismatch')
            if c['rejected_mids']!=sum(c[x] for x in ('probes','blocked_cur','blocked_gap','blocked_budget','blocked_selection')):
                raise ValueError('Gate accounting mismatch')
            if c['ndc_nn']>p['nn_cap'] or c['ndc_nn']>c['probes']*p['nn_limit'] or c['probes']>c['expanded']*p['mid_cap']:
                raise ValueError('Algorithm budget exceeded')
            if c['successful_probes']>min(c['probes'],c['admitted_nn']) or c['admitted_nn']>c['ndc_nn'] or c['ndc_nn']>c['scanned_nn']:
                raise ValueError('Invalid probe accounting')
    if set(groups)!=expected or set(qgroups)!=expected:raise ValueError('Missing measurement points')
    result=[]
    for key in sorted(expected):
        t,q=groups[key],qgroups[key]
        if len(t)!=p['rounds'] or len(q)!=p['limit']:raise ValueError('Missing round/query')
        recall=statistics.mean(float(x['recall']) for x in q);ndc=statistics.mean(int(x['ndc']) for x in q)
        if any(not math.isclose(float(x['recall']),recall,abs_tol=1e-12) or not math.isclose(float(x['mean_ndc']),ndc,abs_tol=1e-9) for x in t):
            raise ValueError('Timing/query results mismatch')
        row=dict(case=case['name'],method=key[0],ef=key[1],recall=recall,mean_ndc=ndc,
            qps_median=statistics.median(float(x['qps']) for x in t),qps_min=min(float(x['qps']) for x in t),qps_max=max(float(x['qps']) for x in t))
        result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--indir',type=Path,required=True);a=ap.parse_args()
    m=json.loads((a.indir/'manifest.json').read_text(encoding='utf-8'));p=m['parameters']
    if not m['completed']:raise ValueError('Experiment is incomplete')
    points=[];mechanism=[]
    for case in m['cases']:
        folder=a.indir/case['name'];q=read(folder/'timings.csv.queries.csv')
        case_points=summarize_case(case,read(folder/'timings.csv'),q,p)
        if p['timing_protocol']=='original':
            attach_latencies(case_points,read(folder/'timings.csv.latencies.csv'),q,p)
        points+=case_points
        for ef in case['other_efs']:
            rows=[r for r in q if r['method']=='selective_prefix' and int(r['ef'])==ef]
            if not rows:continue
            sums={key:sum(int(r[key]) for r in rows) for key in ('ndc_nn','rejected_mids','blocked_cur','blocked_gap','blocked_budget','blocked_selection','probes','successful_probes','admitted_nn','scanned_nn','expanded')}
            mechanism.append(dict(case=case['name'],ef=ef,queries=len(rows),**{key+'_mean':v/len(rows) for key,v in sums.items()},
                successful_probe_rate=sums['successful_probes']/sums['probes'] if sums['probes'] else 0,
                probe_per_rejected_mid=sums['probes']/sums['rejected_mids'] if sums['rejected_mids'] else 0))
    anchor='control' if any(r['case']=='control' for r in points) else 'caps_only'
    baseline=[r for r in points if r['case']==anchor and r['method']=='beam'];matches=[]
    for case in m['cases']:
        for target in (.90,.95,.97,.98,.99):
            b=[r for r in baseline if r['recall']>=target]
            s=[r for r in points if r['case']==case['name'] and r['method']==case['other_method'] and r['recall']>=target]
            b=max(b,key=lambda r:r['qps_median']) if b else None;s=max(s,key=lambda r:r['qps_median']) if s else None
            row=dict(case=case['name'],target_recall=target,beam_reached=bool(b),method_reached=bool(s))
            for label,r in [('beam',b),('method',s)]:
                for key in ('ef','recall','qps_median','mean_ndc'):row[label+'_'+key]=r[key] if r else ''
            row['method_over_beam_qps']=s['qps_median']/b['qps_median'] if s and b else ''
            row['method_over_beam_ndc']=s['mean_ndc']/b['mean_ndc'] if s and b else ''
            row['needs_denser_ef_grid']=abs(s['recall']-b['recall'])>.001 if s and b else ''
            matches.append(row)
    out=a.indir/'analysis';out.mkdir(exist_ok=True)
    write(out/'points.csv',points);write(out/'mechanism.csv',mechanism);write(out/'target_recall.csv',matches)
    (out/'README.md').write_text('Timing includes all search/selection work; counters are collected in a separate pass.\n'
        'prefix means first fresh distinct adjacency IDs, not nearest vectors.\n'
        'successful probe means at least one admitted candidate, not final GT gain.\n'
        'Target rows select measured points reaching the target; unequal attained recall is flagged.\n'
        'No local experiment result is implied by source changes or a dry-run.\n',encoding='utf-8')
    print('Summaries:',out)


if __name__=='__main__':main()
