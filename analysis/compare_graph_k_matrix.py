#!/usr/bin/env python3
"""Compare four completed graph/K runs, rejecting incompatible configurations."""
import argparse
import json
from pathlib import Path
import statistics
import struct
from analyze_hnsw_stage1 import read_csv, write_csv, summarize


def check_pair(a,b):
    for m,label in ((a,'unrepaired_bottom'),(b,'repaired_reference')):
        if m.get('suite')!='hnsw_stage1' or not m.get('completed') or m['graph']!=label:
            raise ValueError('Need completed and correctly labeled runs')
    for key in ('profile','binary_sha256','platform'):
        if a[key]!=b[key]:raise ValueError('Configuration mismatch: '+key)
    if a['parameters'].get('timing_protocol','batch')!=b['parameters'].get('timing_protocol','batch'):
        raise ValueError('Timing protocol mismatch')
    for key in ('k','offset','limit','efs','rounds','warmup_passes','min_batch_ms','seed','verify_limit','cpu'):
        if a['parameters'][key]!=b['parameters'][key]:raise ValueError('Configuration mismatch: '+key)
    for key in ('query','gt'):
        if a['inputs'][key]['sha256']!=b['inputs'][key]['sha256']:
            raise ValueError('Different input contents: '+key)
    ha=struct.unpack('<7Q',bytes.fromhex(a['inputs']['index']['header_hex']))
    hb=struct.unpack('<7Q',bytes.fromhex(b['inputs']['index']['header_hex']))
    if ha!=hb:raise ValueError('Index headers differ')


def target_table(curves,targets):
    result=[]
    for target in targets:
        best={};row={'target_recall':target}
        for graph in ('hnsw','ngfix'):
            for method in ('beam','bound_unbounded'):
                label=graph+'_'+method
                eligible=[p for p in curves if p['graph']==graph and p['method']==method and p['recall']>=target-1e-12]
                point=max(eligible,key=lambda p:p['qps_median']) if eligible else None
                best[label]=point;row[label+'_reached']=point is not None
                for key in ('ef','recall','qps_median','mean_ndc'):
                    row[label+'_'+key]=point[key] if point else ''
        reference=best['hnsw_beam']
        for label,point in best.items():
            row[label+'_qps_over_hnsw_beam']=point['qps_median']/reference['qps_median'] if point and reference else ''
        ours,ngfix=best['hnsw_bound_unbounded'],best['ngfix_beam']
        row['hnsw_bound_qps_over_ngfix_beam']=ours['qps_median']/ngfix['qps_median'] if ours and ngfix else ''
        attained=[p['recall'] for p in best.values() if p]
        row['attained_recall_span_pp']=100*(max(attained)-min(attained)) if attained else ''
        row['needs_denser_ef_grid']=max(attained)-min(attained)>.001 if attained else ''
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir',type=Path,required=True)
    p.add_argument('--targets',default='.90,.95,.97,.98,.99,.995')
    a=p.parse_args();targets=[float(v) for v in a.targets.split(',')]
    if not targets or any(not 0<=v<=1 for v in targets):raise ValueError('Invalid targets')
    out=a.indir/'comparison';out.mkdir(exist_ok=True)
    all_curves=[];mechanisms=[]
    for k in (10,100):
        manifests={};query_rows={};curves=[]
        for graph in ('hnsw','ngfix'):
            folder=a.indir/(graph+'_k'+str(k))
            manifests[graph]=json.loads((folder/'clean/manifest.json').read_text(encoding='utf-8'))
            if manifests[graph]['parameters']['k']!=k:raise ValueError('Wrong K directory')
            query_rows[graph]=read_csv(folder/'clean/timings.csv.queries.csv')
            points=summarize(read_csv(folder/'clean/timings.csv'),query_rows[graph],manifests[graph])
            curves.extend(dict(k=k,graph=graph,**point) for point in points)
            summary=folder/'diagnostics/summary/paired_mechanism_summary.csv'
            if summary.is_file():mechanisms.extend(dict(k=k,graph=graph,**r) for r in read_csv(summary))
        check_pair(manifests['hnsw'],manifests['ngfix'])
        # Require the wrapper's full structural verification, not merely matching headers.
        for graph in ('hnsw','ngfix'):
            report=json.loads((a.indir/(graph+'_k'+str(k))/'index_pair_verification.json').read_text(encoding='utf-8'))
            if not report.get('verified'):raise ValueError('Index pair verification missing/failed')
            if Path(report['base_path']).resolve()!=Path(manifests['hnsw']['inputs']['index']['path']).resolve() or \
               Path(report['repaired_path']).resolve()!=Path(manifests['ngfix']['inputs']['index']['path']).resolve():
                raise ValueError('Verification report refers to different index paths')
        all_curves.extend(curves)
        write_csv(out/('k'+str(k)+'_at_target_recall.csv'),target_table(curves,targets))
        efs=manifests['hnsw']['parameters']['efs'];anchor=200 if 200 in efs else efs[0]
        baseline=[r for r in query_rows['hnsw'] if r['method']=='beam' and int(r['ef'])==anchor]
        baseline.sort(key=lambda r:(float(r['recall']),-int(r['ndc']),int(r['query_id'])))
        import math
        difficult={int(r['query_id']) for r in baseline[:max(1,math.ceil(len(baseline)*.2))]}
        group_rows=[]
        for graph in ('hnsw','ngfix'):
            for method in ('beam','bound_unbounded'):
                for ef in efs:
                    selected=[r for r in query_rows[graph] if r['method']==method and int(r['ef'])==ef and int(r['query_id']) in difficult]
                    group_rows.append(dict(k=k,graph=graph,method=method,ef=ef,anchor_ef=anchor,queries=len(selected),
                                           recall=statistics.mean(float(r['recall']) for r in selected),
                                           mean_ndc=statistics.mean(int(r['ndc']) for r in selected)))
        write_csv(out/('k'+str(k)+'_fixed_hnsw_hard_queries.csv'),group_rows)
    write_csv(out/'all_recall_qps_ndc_curves.csv',all_curves)
    write_csv(out/'all_mechanism_summaries.csv',mechanisms)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Comparison CSVs ready:',out);return
    for k in (10,100):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for graph in ('hnsw','ngfix'):
            for method in ('beam','bound_unbounded'):
                curve=sorted([r for r in all_curves if r['k']==k and r['graph']==graph and r['method']==method],key=lambda r:r['recall'])
                for ax,key in zip(axes,('qps_median','mean_ndc')):
                    ax.plot([r['recall'] for r in curve],[r[key] for r in curve],'-o',label=graph+' '+method)
                    ax.set_xlabel('Recall@'+str(k));ax.set_ylabel(key);ax.grid(alpha=.3);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/('k'+str(k)+'_recall_qps_ndc.png'),dpi=180);plt.close(fig)
    print('Comparison:',out)


if __name__=='__main__':main()
