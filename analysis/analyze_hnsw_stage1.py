#!/usr/bin/env python3
"""Validate stage-1 measurements, summarize QPS/NDC, and match attained recall."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantile(values, fraction):
    values=sorted(values)
    pos=(len(values)-1)*fraction
    lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def summarize(rows, queries, m):
    p=m['parameters']; groups={}; seen=set()
    expected={(method,ef) for method in ('beam','bound_unbounded') for ef in p['efs']}
    for r in rows:
        key=(r['method'],int(r['ef'])); round_=int(r['round'])
        if key not in expected or (*key,round_) in seen or not 0<=round_<p['rounds']:
            raise ValueError('Unexpected or duplicate timing point')
        seen.add((*key,round_))
        if (int(r['k']),int(r['queries']),int(r['query_offset'])) != (p['k'],p['limit'],p['offset']):
            raise ValueError('Query/K mismatch')
        if int(r['verified_queries']) != min(p['verify_limit'],p['limit']):
            raise ValueError('Wrong reference verification population')
        qps,sec,recall,ndc=(float(r[k]) for k in ('qps','seconds','recall','mean_ndc'))
        batch=int(r['batch_queries'])
        if any(not math.isfinite(v) for v in (qps,sec,recall,ndc)) or min(qps,sec,ndc)<=0 or not 0<=recall<=1:
            raise ValueError('Invalid measurements')
        original=p.get('timing_protocol','batch')=='original'
        if r.get('timing_protocol','batch')!=p.get('timing_protocol','batch'):
            raise ValueError('Mixed timing protocols')
        if batch<p['limit'] or batch%p['limit'] or not math.isclose(qps,batch/sec,rel_tol=1e-10) or \
           (not original and sec*1000+1e-6<p['min_batch_ms']) or (original and batch!=p['limit']):
            raise ValueError('Invalid batch timing accounting')
        groups.setdefault(key,[]).append((qps,recall,ndc))
    qgroups={}; qseen=set()
    for r in queries:
        key=(r['method'],int(r['ef']));qid=int(r['query_id'])
        recall,ndc=float(r['recall']),int(r['ndc'])
        if key not in expected or (*key,qid) in qseen or not p['offset']<=qid<p['offset']+p['limit']:
            raise ValueError('Duplicate or unexpected query statistics')
        if not math.isfinite(recall) or not 0<=recall<=1 or ndc<1:
            raise ValueError('Invalid per-query statistics')
        qseen.add((*key,qid));qgroups.setdefault(key,[]).append((recall,ndc))
    if set(groups)!=expected or set(qgroups)!=expected:
        raise ValueError('Missing points')
    points=[]
    for key in sorted(expected):
        timings=groups[key]; qs=qgroups[key]
        if len(timings)!=p['rounds'] or len(qs)!=p['limit']:
            raise ValueError('Missing rounds or queries')
        recall=statistics.mean(v[0] for v in qs);ndc=statistics.mean(v[1] for v in qs)
        if any(not math.isclose(v[1],recall,abs_tol=1e-12) or not math.isclose(v[2],ndc,abs_tol=1e-9) for v in timings):
            raise ValueError('Query/timing recall or NDC mismatch')
        speeds=[v[0] for v in timings]
        points.append(dict(method=key[0],ef=key[1],recall=recall,mean_ndc=ndc,
                           ndc_p50=quantile([v[1] for v in qs],.5),ndc_p95=quantile([v[1] for v in qs],.95),
                           ndc_p99=quantile([v[1] for v in qs],.99),qps_median=statistics.median(speeds),
                           qps_min=min(speeds),qps_max=max(speeds),qps_q25=quantile(speeds,.25),qps_q75=quantile(speeds,.75)))
    return points


def match_targets(points, targets):
    result=[]
    for target in targets:
        row={'target_recall':target}; best={}
        for method in ('beam','bound_unbounded'):
            eligible=[v for v in points if v['method']==method and v['recall']>=target-1e-12]
            best[method]=max(eligible,key=lambda v:v['qps_median']) if eligible else None
            row[method+'_reached']=bool(eligible)
            for key in ('ef','recall','qps_median','mean_ndc'):
                row[method+'_'+key]=best[method][key] if eligible else ''
        a,b=best['beam'],best['bound_unbounded']
        row['bound_over_beam_qps']=b['qps_median']/a['qps_median'] if a and b else ''
        row['bound_over_beam_ndc']=b['mean_ndc']/a['mean_ndc'] if a and b else ''
        row['bound_minus_beam_recall_pp']=100*(b['recall']-a['recall']) if a and b else ''
        row['needs_denser_ef_grid']=abs(b['recall']-a['recall'])>.001 if a and b else ''
        result.append(row)
    return result


def paired_groups(queries, anchor_ef):
    lookup={(r['method'],int(r['ef']),int(r['query_id'])):r for r in queries}
    anchors=[r for r in queries if r['method']=='beam' and int(r['ef'])==anchor_ef]
    anchors.sort(key=lambda r:(float(r['recall']),-int(r['ndc']),int(r['query_id'])))
    difficult={int(r['query_id']) for r in anchors[:max(1,math.ceil(len(anchors)*.2))]}
    ids={int(r['query_id']) for r in anchors};rows=[]
    for ef in sorted({int(r['ef']) for r in queries}):
        for label,qids in [('all',ids),('fixed_low_recall_20pct',difficult),('remaining_80pct',ids-difficult)]:
            if not qids: continue
            pairs=[(lookup['beam',ef,q],lookup['bound_unbounded',ef,q]) for q in qids]
            deltas=[float(b['recall'])-float(a['recall']) for a,b in pairs]
            rows.append(dict(ef=ef,group=label,anchor_ef=anchor_ef,queries=len(pairs),
                             beam_recall=statistics.mean(float(a['recall']) for a,b in pairs),
                             bound_recall=statistics.mean(float(b['recall']) for a,b in pairs),
                             delta_recall=statistics.mean(deltas),
                             improved_fraction=sum(v>1e-12 for v in deltas)/len(deltas),
                             worsened_fraction=sum(v< -1e-12 for v in deltas)/len(deltas),
                             mean_delta_ndc=statistics.mean(int(b['ndc'])-int(a['ndc']) for a,b in pairs)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir',type=Path,required=True)
    p.add_argument('--targets',default='.90,.95,.97,.98,.99,.995')
    p.add_argument('--anchor-ef',type=int,default=200)
    a=p.parse_args();m=json.loads((a.indir/'manifest.json').read_text(encoding='utf-8'))
    if m.get('suite')!='hnsw_stage1' or not m.get('completed'):
        raise ValueError('Need a completed hnsw_stage1 sweep')
    queries=read_csv(a.indir/'timings.csv.queries.csv')
    points=summarize(read_csv(a.indir/'timings.csv'),queries,m)
    targets=[float(v) for v in a.targets.split(',')]
    if any(not math.isfinite(v) or not 0<=v<=1 for v in targets):raise ValueError('Invalid target')
    out=a.indir/'analysis';out.mkdir(exist_ok=True)
    write_csv(out/'recall_qps_ndc_curve.csv',points)
    write_csv(out/'at_target_recall.csv',match_targets(points,targets))
    anchor=a.anchor_ef if a.anchor_ef in m['parameters']['efs'] else m['parameters']['efs'][0]
    write_csv(out/'paired_query_groups.csv',paired_groups(queries,anchor))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('CSV results ready; install matplotlib only if plots are wanted:',out)
        return
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('beam','bound_unbounded'):
        curve=sorted([v for v in points if v['method']==method],key=lambda v:v['recall'])
        for ax,key in zip(axes,('qps_median','mean_ndc')):
            ax.plot([v['recall'] for v in curve],[v[key] for v in curve],'-o',label=method)
            ax.set_xlabel('Recall@'+str(m['parameters']['k']));ax.set_ylabel(key);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(out/'recall_qps_ndc.png',dpi=180);plt.close(fig)
    print('Results:',out)


if __name__=='__main__':main()
