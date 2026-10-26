#!/usr/bin/env python3
"""Compare median single-worker QPS at measured Recall thresholds; no interpolation."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(rows, manifest):
    p=manifest['parameters']; groups={}; keys=set()
    expected={('beam',ef) for ef in p['beam_efs']}|{('cheap32_nn8',ef) for ef in p['cheap_efs']}
    for raw in rows:
        method=raw['method']; ef=int(raw['ef']); round_=int(raw['round'])
        key=(method,ef,round_)
        if key in keys: raise ValueError('Duplicate method/ef/round')
        keys.add(key)
        if (method,ef) not in expected or not 0<=round_<p['rounds']: raise ValueError('Unexpected timing point')
        if int(raw['k'])!=p['k'] or int(raw['query_offset'])!=p['offset'] or int(raw['queries'])!=p['limit']:
            raise ValueError('Different query population/K')
        if int(raw['verified_queries'])!=min(p.get('verify_limit',p['limit']),p['limit']):
            raise ValueError('Reference verification count differs from requested count')
        seconds=float(raw['seconds']); qps=float(raw['qps']); recall=float(raw['recall'])
        queries=int(raw['batch_queries'])
        if any(not math.isfinite(v) for v in [seconds,qps,recall]) or min(seconds,qps)<=0 or not 0<=recall<=1:
            raise ValueError('Invalid timing/recall value')
        if queries<p['limit'] or queries%p['limit'] or not math.isclose(qps,queries/seconds,rel_tol=1e-10):
            raise ValueError('Batch QPS accounting mismatch')
        if seconds*1000+1e-6<p['min_batch_ms']: raise ValueError('Batch too short')
        groups.setdefault((method,ef),[]).append((recall,qps))
    if set(groups)!=expected: raise ValueError('Missing timing points')
    result=[]
    for (method,ef),values in sorted(groups.items()):
        if len(values)!=p['rounds']: raise ValueError('Missing timing rounds')
        recalls=[v[0] for v in values]
        if max(recalls)-min(recalls)>1e-12: raise ValueError('Recall changed between rounds')
        qps=sorted(v[1] for v in values)
        quartiles=statistics.quantiles(qps,n=4,method='inclusive') if len(qps)>1 else [qps[0]]*3
        median=statistics.median(qps)
        result.append(dict(method=method,ef=ef,queries=p['limit'],rounds=p['rounds'],recall=recalls[0],
            qps_median=median,qps_min=min(qps),qps_max=max(qps),qps_q25=quartiles[0],qps_q75=quartiles[2],
            equivalent_mean_query_us=1e6/median))
    return result


def at_targets(points,targets):
    rows=[]
    for target in targets:
        best={}
        for method in ['beam','cheap32_nn8']:
            candidates=[r for r in points if r['method']==method and r['recall']>=target-1e-12]
            best[method]=max(candidates,key=lambda r:r['qps_median']) if candidates else None
        row=dict(target_recall=target)
        for method,label in [('beam','beam'),('cheap32_nn8','cheap')]:
            point=best[method]
            row[label+'_reached']=point is not None
            for key in ['ef','recall','qps_median']:
                row[label+'_'+key]='' if point is None else point[key]
        both=all(best.values())
        row['qps_ratio_cheap_over_beam']=best['cheap32_nn8']['qps_median']/best['beam']['qps_median'] if both else ''
        row['recall_delta_pp']=100*(best['cheap32_nn8']['recall']-best['beam']['recall']) if both else ''
        rows.append(row)
    return rows


def match_cheap_points(points):
    rows=[]
    for cheap in [r for r in points if r['method']=='cheap32_nn8']:
        candidates=[r for r in points if r['method']=='beam' and r['recall']>=cheap['recall']-1e-12]
        beam=max(candidates,key=lambda r:r['qps_median']) if candidates else None
        rows.append(dict(cheap_ef=cheap['ef'],cheap_recall=cheap['recall'],cheap_qps_median=cheap['qps_median'],
            beam_reached=beam is not None,beam_ef='' if beam is None else beam['ef'],
            beam_recall='' if beam is None else beam['recall'],beam_qps_median='' if beam is None else beam['qps_median'],
            qps_ratio_cheap_over_beam='' if beam is None else cheap['qps_median']/beam['qps_median'],
            beam_recall_extra_pp='' if beam is None else 100*(beam['recall']-cheap['recall'])))
    return rows


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--indir',type=Path,required=True)
    p.add_argument('--targets',default='0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.992,0.995')
    a=p.parse_args(); m=json.loads((a.indir/'manifest.json').read_text(encoding='utf-8'))
    if m.get('suite')!='clean_qps' or not m.get('completed'): raise ValueError('Need a completed clean QPS sweep')
    with (a.indir/'timings.csv').open(newline='',encoding='utf-8') as f: points=summarize(list(csv.DictReader(f)),m)
    targets=[float(v) for v in a.targets.split(',')]
    if not targets or any(not math.isfinite(v) or not 0<=v<=1 for v in targets): raise ValueError('Invalid Recall targets')
    matched=at_targets(points,targets)
    out=a.indir/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'recall_qps_curve.csv',points);write_csv(out/'at_target_recall_qps.csv',matched)
    write_csv(out/'matched_to_cheap_recall.csv',match_cheap_points(points))
    print('Measured Recall/QPS curves (median of rounds):')
    for row in points: print(f"{row['method']:13s} ef={row['ef']:4d} Recall={row['recall']:.5f} QPS={row['qps_median']:.2f}")
    print('\nMeasured points reaching each target; ratio >1 favors cheap32, <1 favors beam:')
    for row in matched:
        ratio=row['qps_ratio_cheap_over_beam']
        print(f"target={row['target_recall']:.4f} beam_ef={row['beam_ef']} cheap_ef={row['cheap_ef']} ratio={ratio}")
    print('No Recall interpolation/tolerance; actual recalls are shown. Cold start and multi-worker throughput are not measured.')

if __name__=='__main__': main()
