"""Plot the measured NGFix residual-query comparisons, without new benchmarks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/residual_query_analysis_20261014'
q=pd.read_csv(ROOT/'result/complete_experiment_report_20261013/tables/performance_queries.csv')
d=pd.read_csv(OUT/'ngfix_recall_deciles.csv')
plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':170})
fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
for ax,(k,ef) in zip(axes,[(10,20),(10,100),(100,400)]):
    f=d[(d.k==k)&(d.a_ef==ef)]
    h=f[f.a_graph=='hnsw'].sort_values('decile_hardest_first')
    n=f[f.a_graph=='ngfix'].sort_values('decile_hardest_first')
    x=h.decile_hardest_first
    ax.plot(x,h.b_recall,'o-',label=f'NGFix beam ef{ef}',color='#315a83')
    ax.plot(x,h.a_recall,'s-',label=f'HNSW twohop ef{ef}',color='#c17732')
    ax.plot(x,n.a_recall,'^-',label=f'NGFix twohop ef{ef}',color='#3a8c64')
    if (k,ef)==(10,20):
        anchor=q[(q.graph=='ngfix')&(q.k==k)&(q.method=='beam')&(q.ef==ef)].sort_values(['recall','ndc','query_id'],ascending=[True,False,True])
        labels={qid:i//100+1 for i,qid in enumerate(anchor.query_id)}
        other=q[(q.graph=='ngfix')&(q.k==k)&(q.method=='beam')&(q.ef==30)].copy()
        other['decile']=other.query_id.map(labels)
        ax.plot(range(1,11),other.groupby('decile').recall.mean(),'d--',label='NGFix beam ef30',color='#74629d')
    ax.set_title(f'K={k}, fixed NGFix beam ef={ef} cohorts')
    ax.set_xticks(range(1,11));ax.set_ylim(0,1.025);ax.grid(alpha=.18)
    ax.set_xlabel('NGFix recall decile (1 = lowest recall)')
    ax.legend(fontsize=8,loc='lower right')
axes[0].set_ylabel('Mean recall on the SAME query IDs')
fig.suptitle('A lowest-recall cohort is an offline GT grouping, not an online selector',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(OUT/'ngfix_difficulty_deciles.png');fig.savefig(OUT/'ngfix_difficulty_deciles.svg');plt.close(fig)

m=pd.read_csv(OUT/'mechanism_missing_error_closed.csv')
order=['hnsw_k10_ef20','hnsw_k10_ef100','hnsw_k100_ef100','hnsw_k100_ef400',
       'ngfix_k10_ef20','ngfix_k10_ef100','ngfix_k100_ef100','ngfix_k100_ef400']
m=m.set_index('case').loc[order]
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
y=np.arange(8);labels=[s.replace('_',' / ') for s in order]
axes[0].barh(y,100*m.net_fraction_of_missing_closed,color=['#c17732']*4+['#3a8c64']*4)
axes[0].set_xlabel('Net fraction of paired beam missed GT closed (%)');axes[0].set_xlim(0,100)
axes[1].barh(y,m.extra_ndc_per_net_gt,color=['#c17732']*4+['#3a8c64']*4)
axes[1].set_xscale('log');axes[1].set_xlabel('Extra full distances per net additional GT (log scale)')
axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
fig.suptitle('Residual errors are recoverable, but current unbounded search is expensive')
fig.tight_layout();fig.savefig(OUT/'residual_recovery_and_cost.png');fig.savefig(OUT/'residual_recovery_and_cost.svg');plt.close(fig)
print('Created 2 PNG and 2 SVG figures in',OUT)
