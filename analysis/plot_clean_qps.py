#!/usr/bin/env python3
"""Render measured Recall/QPS curves and threshold-matched QPS ratios."""
import argparse
import csv
import json
from pathlib import Path
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as error:
    raise SystemExit('Plotting needs matplotlib. Existing timing CSVs remain valid. Install it, then rerun this plotting script.') from error


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--indir',type=Path,required=True)
    a=p.parse_args();out=a.indir/'analysis'
    meta=json.loads((a.indir/'manifest.json').read_text(encoding='utf-8'))
    k=meta['parameters']['k']
    with (out/'recall_qps_curve.csv').open(newline='',encoding='utf-8') as f: points=list(csv.DictReader(f))
    with (out/'at_target_recall_qps.csv').open(newline='',encoding='utf-8') as f: targets=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8.5,5.5),layout='constrained')
    for method,label,color,marker in [('beam','NGFix beam','#333333','o'),('cheap32_nn8','Cheap32, NN8','#2469b4','D')]:
        rows=sorted([r for r in points if r['method']==method],key=lambda r:int(r['ef']))
        if not rows: raise ValueError('Missing curve: '+method)
        x=[100*float(r['recall']) for r in rows];y=[float(r['qps_median']) for r in rows]
        ax.plot(x,y,color=color,marker=marker,markersize=4,linewidth=1.5,label=label)
        ax.fill_between(x,[float(r['qps_q25']) for r in rows],[float(r['qps_q75']) for r in rows],color=color,alpha=.13)
        for r in rows:
            if int(r['ef']) in [100,150,200,300,400,600,1000]:
                ax.annotate('ef='+r['ef'],(100*float(r['recall']),float(r['qps_median'])),
                            xytext=(5,5 if method=='beam' else -12),textcoords='offset points',fontsize=8,color=color)
    ax.set_xlabel(f'Recall@{k} (%)');ax.set_ylabel('Single-worker QPS (median of rounds)')
    ax.set_title('Measured Recall / QPS; shaded bands are round-to-round IQR')
    ax.grid(alpha=.22);ax.legend();ax.set_ylim(bottom=0);ax.margins(x=.09)
    if meta.get('synthetic_test_only'): fig.suptitle('Synthetic test data; NOT measured ANN results',fontsize=11)
    for extension in ['png','svg']:fig.savefig(out/('recall_qps.'+extension),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    valid=[r for r in targets if r['qps_ratio_cheap_over_beam']!='']
    if valid:
        ax.plot([100*float(r['target_recall']) for r in valid],
                [float(r['qps_ratio_cheap_over_beam']) for r in valid],marker='o',linestyle='None',color='#2469b4')
    else:ax.text(.5,.5,'No target reached by both methods',ha='center',va='center',transform=ax.transAxes)
    ax.axhline(1,color='#666666',linestyle='--',linewidth=1)
    ax.set_xlabel(f'Target Recall@{k} (%)');ax.set_ylabel('QPS ratio: cheap32 / beam')
    ax.set_title('Fastest measured points reaching each target; above 1 favors cheap32')
    ax.grid(alpha=.22)
    fig.text(.5,.018,'Actual recalls may differ; see at_target_recall_qps.csv. No interpolation.',ha='center',fontsize=8)
    if meta.get('synthetic_test_only'): fig.suptitle('Synthetic test data; NOT measured ANN results',fontsize=11)
    fig.tight_layout(rect=(0,.08,1,.94 if meta.get('synthetic_test_only') else 1))
    for extension in ['png','svg']:fig.savefig(out/('matched_recall_qps.'+extension),dpi=180)
    plt.close(fig)
    print('Plots:',out/'recall_qps.png',out/'matched_recall_qps.png')

if __name__=='__main__': main()
