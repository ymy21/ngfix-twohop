"""Scientific PNG/SVG figures from consolidated full-precision tables.
Matplotlib is used from the available host environment because it is absent
from the bundled Python runtime. Analysis remains in the bundled runtime.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/complete_experiment_report_20261013'
T=OUT/'tables';F=OUT/'figures';F.mkdir(exist_ok=True)
ORDER=['hnsw_k10_ef20','hnsw_k10_ef100','hnsw_k100_ef100','hnsw_k100_ef400',
       'ngfix_k10_ef20','ngfix_k10_ef100','ngfix_k100_ef100','ngfix_k100_ef400']
LABELS=[x.replace('_',' / ') for x in ORDER]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    fig.tight_layout();fig.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def plots():
    p=pd.read_csv(T/'performance_all_points.csv')
    for k in [10,100]:
        fig,ax=plt.subplots(1,3,figsize=(17,4.5))
        for graph,color in [('hnsw','#2565a5'),('ngfix','#d5672c')]:
            for method,style in [('beam','-'),('bound_unbounded','--')]:
                g=p[(p.k==k)&(p.graph==graph)&(p.method==method)].sort_values('ef')
                label=graph.upper()+(' beam' if method=='beam' else ' bound-twohop')
                for a,y in zip(ax,['qps_median','mean_ndc','latency_p95_us']):
                    a.plot(g.recall,g[y],style,color=color,marker='o',markersize=3,label=label)
                    a.set_xlabel(f'Recall@{k}');a.set_yscale('log');a.grid(alpha=.2,which='both')
        for a,label in zip(ax,['QPS (higher is better)','Mean NDC (lower is better)','P95 search latency (microseconds)']):a.set_ylabel(label)
        ax[0].legend(fontsize=8);fig.suptitle(f'K={k}: 1,000 queries, one round, original timing protocol')
        save(fig,f'performance_k{k}')
    m=pd.read_csv(T/'mechanism_cases.csv').set_index('case').loc[ORDER]
    fig,ax=plt.subplots(figsize=(11,5))
    left=np.zeros(8)
    for field,label,color in [('ndc_entry_mean','Entry point','#38485e'),('ndc_mid_mean','First-hop','#6a9dc3'),('ndc_nn_nojump_mean','Second-hop, no admission','#b4b8be'),('ndc_nn_jump_mean','Second-hop, >=1 admission','#e8a25b')]:
        values=100*m[field].to_numpy()/m.ndc_mean.to_numpy();ax.barh(LABELS,values,left=left,label=label,color=color);left+=values
    ax.invert_yaxis();ax.set_xlabel('Share of total query NDC (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
    ax.set_title('Distance cost is dominated by probes admitting no candidate')
    save(fig,'ndc_cost_decomposition')
    o=pd.read_csv(T/'probe_outcomes.csv');fig,axes=plt.subplots(1,2,figsize=(15,5))
    colors=['#e0e3e7','#b4b8be','#e8a25b','#3a9669']
    names=['FAILED_NO_FRESH','FAILED_WITH_FRESH','SUCCESS_NO_RECOVERED_GT','SUCCESS_RECOVERED_GT']
    for a,value in zip(axes,['probes','nn_ndc']):
        table=o.pivot(index='case',columns='group',values=value).reindex(ORDER).fillna(0)
        left=np.zeros(8)
        for label,color in zip(names,colors):
            v=100*table[label]/table.sum(axis=1);a.barh(LABELS,v,left=left,color=color,label=label);left+=v
        a.invert_yaxis();a.set_xlabel('Share (%)');a.set_title('Probe count' if value=='probes' else 'Second-hop NDC')
    axes[1].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2);fig.suptitle('Nearest-root attribution: "no recovered GT" is not a causal deletion label')
    save(fig,'probe_outcomes')
    fig,ax=plt.subplots(figsize=(11,5))
    direct=100*m.recovered_gt_direct_jump_mean/m.k;routing=100*m.recovered_gt_bridge_descendant_mean/m.k;lost=100*m.lost_gt_total_mean/m.k
    ax.barh(LABELS,direct,color='#3a9669',label='Direct recovered GT');ax.barh(LABELS,routing,left=direct,color='#6a9dc3',label='Routing recovered GT')
    ax.barh(LABELS,-lost,color='#c65e58',label='Lost GT');ax.axvline(0,color='#555',linewidth=.7);ax.invert_yaxis()
    ax.set_xlabel('Recall percentage points');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3);ax.set_title('Recovered and lost GT vs same-graph, same-ef beam')
    save(fig,'gt_recovery_and_loss')
    trend=pd.read_csv(T/'feature_trends.csv')
    fields=['mid_gap_normalized','cur_slack_normalized','state_steps_since_topk_update','state_step','state_discovery_depth','mid_total_degree','state_frontier_size','topk_spread']
    matrix=np.zeros((8,len(fields)))
    for i,case in enumerate(ORDER):
        for j,field in enumerate(fields):
            r=trend[(trend['case']==case)&(trend.feature==field)].iloc[0]
            matrix[i,j]=np.log2(r.high_useful/r.low_useful) if r.high_useful and r.low_useful else np.nan
    fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(matrix,cmap='RdBu',vmin=-5,vmax=5,aspect='auto')
    ax.set_yticks(range(8),LABELS);ax.set_xticks(range(len(fields)),fields,rotation=25,ha='right',fontsize=8)
    for i in range(8):
        for j in range(len(fields)):ax.text(j,i,f'{2**matrix[i,j]:.2f}x',ha='center',va='center',fontsize=8,color='white' if abs(matrix[i,j])>2.8 else '#222')
    fig.colorbar(im,ax=ax,label='log2(high-value-bin useful rate / low-value-bin useful rate)')
    ax.set_title('Feature associations: red favors low values; blue favors high values')
    save(fig,'feature_associations')
    phase=pd.read_csv(T/'search_phase_bins.csv');fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for case in ORDER:
        g=phase[phase['case']==case].sort_values('bin');x=(g.bin+.5)/20
        axes[0].plot(x,g.useful/g.probes.replace(0,np.nan),label=case)
        axes[1].plot(x,1000*(g.direct_new+g.recovered_desc)/g.nn_ndc.replace(0,np.nan),label=case)
    axes[0].set_yscale('log');axes[1].set_yscale('log');axes[0].set_ylabel('Useful probe fraction');axes[1].set_ylabel('Attributed recovered GT / 1k NN distances')
    axes[1].set_xlabel('Search position normalized by FINAL expanded count (offline only)');axes[0].legend(fontsize=7,ncol=2)
    save(fig,'search_stage_yield')
    paths=pd.read_csv(T/'baseline_path_groups.csv');fig,ax=plt.subplots(figsize=(11,5));left=np.zeros(8)
    for group,label,color in [('SHARED_CUR_MID_NOT_EXPANDED','Baseline expanded cur, did not expand mid','#6a9dc3'),('NEW_UPSTREAM_CUR','Baseline did not expand cur','#e8a25b'),('BASELINE_MID_EXPANDED','Baseline expanded mid','#b4b8be')]:
        lookup=paths.set_index(['case','group']);vals=np.array([lookup.loc[(c,group)].recovered_gt for c in ORDER],float)
        totals=paths.groupby('case').recovered_gt.sum().loc[ORDER].to_numpy();v=100*vals/totals
        ax.barh(LABELS,v,left=left,label=label,color=color);left+=v
    ax.invert_yaxis();ax.set_xlabel('Share of uniquely attributed recovered GT (%)');ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16))
    ax.set_title('Existing blocked path opportunities and new upstream exploration both contribute')
    save(fig,'baseline_path_recovery')
    edge=pd.read_csv(T/'edge_pair_recovery.csv');fig,axes=plt.subplots(1,2,figsize=(15,5))
    names=['HH','HN','HR','NH','NN','NR','RH','RN','RR'];colors=plt.cm.tab10(np.linspace(0,1,9))
    for ax,field in zip(axes,['fresh_scored_share','attributed_recovered_gt_share']):
        table=edge.pivot(index='case',columns='edge_pair',values=field).reindex(ORDER).fillna(0);left=np.zeros(8)
        for label,color in zip(names,colors):
            v=100*table[label];ax.barh(LABELS,v,left=left,label=label,color=color);left+=v
        ax.invert_yaxis();ax.set_xlabel('Share (%)');ax.set_title('Fresh-scored NN cost' if field=='fresh_scored_share' else 'Unique recovered GT attribution')
    axes[1].legend(ncol=5,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16));fig.suptitle('H: base edge; N: NGFix edge; R: RFix edge')
    save(fig,'edge_pairs_cost_and_recovery')
    sp=pd.read_csv(T/'sparse_jaccard_controls.csv');fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,value,label in zip(axes,['useful_rate','recovered_gt_per_1k_nn_ndc'],['Useful probe fraction','Attributed recovered GT / 1k NN distances']):
        table=sp.pivot(index='case',columns='group',values=value).reindex(ORDER)
        y=np.arange(8);ax.barh(y-.17,table.ZERO,height=.34,label='No common out-neighbor',color='#b4b8be');ax.barh(y+.17,table.POSITIVE,height=.34,label='At least one common out-neighbor',color='#6a9dc3')
        ax.set_yticks(y,LABELS);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel(label);ax.legend(fontsize=7)
    fig.suptitle('Sparse topology: original quartile bins hid the zero/nonzero difference')
    save(fig,'jaccard_zero_nonzero')
    print('Figures:',len(list(F.glob('*.png'))))


if __name__=='__main__':plots()
