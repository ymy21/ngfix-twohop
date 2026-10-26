#!/usr/bin/env python3
"""Reconstruct performance tables from this workflow's pasted console output.
Values have console precision; use original CSVs for final numerical reporting.
"""
import argparse
import json
from pathlib import Path
import re


def parse_log(text):
    pending={};points=[];ratio=None;groups=[]
    for line in text.splitlines():
        m=re.match(r'evaluation (beam|bound_unbounded) ef=(\d+) recall=([0-9.eE+\-]+) mean_ndc=([0-9.eE+\-]+)',line)
        if m:
            method,ef,recall,ndc=m.groups();key=(method,int(ef))
            if key in pending:raise ValueError('Duplicate evaluation point')
            pending[key]=dict(method=method,ef=int(ef),recall=float(recall),mean_ndc=float(ndc))
        m=re.match(r'evaluation-only mean distance ratio=([0-9.eE+\-]+)',line)
        if m:ratio=float(m.group(1))
        m=re.match(r'round=(\d+) (beam|bound_unbounded) ef=(\d+) recall=([0-9.eE+\-]+) QPS=([0-9.eE+\-]+)',line)
        if m:
            rnd,method,ef,recall,qps=m.groups();p=pending[method,int(ef)]
            if 'qps' in p:raise ValueError('This parser requires one timing round')
            if abs(p['recall']-float(recall))>1e-12:raise ValueError('Recall mismatch')
            p.update(round=int(rnd),qps=float(qps),distance_ratio=ratio)
        m=re.search(r'^Results: .*[/\\](hnsw|ngfix)_k(10|100)[/\\]clean[/\\]analysis',line)
        if m:
            graph,k=m.groups();k=int(k)
            if len(pending)!=(40 if k==10 else 32) or any('qps' not in p for p in pending.values()):
                raise ValueError('Incomplete graph/K block')
            points.extend(dict(graph=graph,k=k,**p) for p in pending.values())
            groups.append((graph,k));pending={};ratio=None
    if pending or set(groups)!={('hnsw',10),('hnsw',100),('ngfix',10),('ngfix',100)}:
        raise ValueError('Incomplete or unexpected suite')
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--log',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();points=parse_log(a.log.read_text(encoding='utf-8'));a.out.mkdir(parents=True,exist_ok=True)
    matches=[]
    for bound in [p for p in points if p['method']=='bound_unbounded']:
        candidates=[p for p in points if p['graph']==bound['graph'] and p['k']==bound['k'] and
                    p['method']=='beam' and p['recall']>=bound['recall']]
        beam=max(candidates,key=lambda p:p['qps']) if candidates else None
        matches.append(dict(graph=bound['graph'],k=bound['k'],bound_ef=bound['ef'],bound_recall=bound['recall'],
            comparable=beam is not None,beam_ef=beam['ef'] if beam else None,
            beam_recall=beam['recall'] if beam else None,beam_qps_over_bound=beam['qps']/bound['qps'] if beam else None,
            bound_ndc_over_beam=bound['mean_ndc']/beam['mean_ndc'] if beam else None))
    payload=dict(source=str(a.log),queries=1000,rounds=1,timing_protocol='original',
                 precision='reconstructed from rounded console values',points=points,matched_bound_points=matches)
    (a.out/'performance_data.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# 性能实验整理','', '1000 条相同查询，1 轮，original 逐查询计时。数据由控制台重建，保留日志精度；最终报告优先使用服务器 CSV。','']
    for k in (10,100):
        for graph in ('hnsw','ngfix'):
            lookup={(p['method'],p['ef']):p for p in points if p['k']==k and p['graph']==graph}
            report.extend([f'## {graph.upper()}，K={k}','',
                '| ef | beam recall | beam QPS | beam NDC | 二跳 recall | 二跳 QPS | 二跳 NDC |',
                '|---:|---:|---:|---:|---:|---:|---:|'])
            for ef in sorted({p['ef'] for p in lookup.values()}):
                b,t=lookup['beam',ef],lookup['bound_unbounded',ef]
                report.append(f"| {ef} | {b['recall']:.5f} | {b['qps']:.2f} | {b['mean_ndc']:.1f} | {t['recall']:.5f} | {t['qps']:.2f} | {t['mean_ndc']:.1f} |")
            report.append('')
    report.extend(['## 有重叠 recall 范围的实测对照','',
        '为每个二跳点选择达到至少同样 recall 的最快 beam 实测点；未达到该 recall 的 beam 不外推。','',
        '| 图 | K | 可比较二跳点 | beam 更快的点数 | beam/二跳 QPS 范围 |','|---|---:|---:|---:|---:|'])
    for k in (10,100):
        for graph in ('hnsw','ngfix'):
            group=[m for m in matches if m['k']==k and m['graph']==graph and m['comparable']]
            speeds=[m['beam_qps_over_bound'] for m in group]
            report.append(f"| {graph} | {k} | {len(group)} | {sum(v>1 for v in speeds)} | {min(speeds):.2f}–{max(speeds):.2f} |")
    (a.out/'performance_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for k in (10,100):
        fig,axes=plt.subplots(1,2,figsize=(12,4.7))
        for graph,color in [('hnsw','#2672b6'),('ngfix','#d96822')]:
            for method,style in [('beam','-'),('bound_unbounded','--')]:
                group=sorted([p for p in points if p['k']==k and p['graph']==graph and p['method']==method],key=lambda p:p['ef'])
                label=('HNSW' if graph=='hnsw' else 'NGFix')+(' beam' if method=='beam' else ' bound-twohop')
                for ax,value in zip(axes,('qps','mean_ndc')):
                    ax.plot([p['recall'] for p in group],[p[value] for p in group],style,marker='o',markersize=3,color=color,label=label)
                    ax.set_xlabel(f'Recall@{k}');ax.set_yscale('log');ax.grid(alpha=.2,which='both')
        axes[0].set_ylabel('QPS (higher is better)');axes[1].set_ylabel('Mean NDC (lower is better)')
        axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
        fig.suptitle(f'K={k} | 1,000 queries | one round | original timing protocol')
        fig.tight_layout();fig.savefig(a.out/f'k{k}_recall_qps_ndc.png',dpi=180);plt.close(fig)
    print('Validated points:',len(points))
    for k in (10,100):
        for graph in ('hnsw','ngfix'):
            group=[m for m in matches if m['k']==k and m['graph']==graph and m['comparable']]
            print(graph,k,'comparable',len(group),'beam faster',sum(m['beam_qps_over_bound']>1 for m in group))


if __name__=='__main__':main()
