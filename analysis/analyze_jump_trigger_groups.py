#!/usr/bin/env python3
"""Offline group diagnostics for a trigger sweep; no GT labels enter online search."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize_jump_gates import load_run


def group_metrics(variant, always, baseline):
    if not variant.index.equals(always.index) or not baseline.index.equals(always.index):
        raise ValueError('Group comparisons require identical query keys')
    helpful=always.recall.gt(baseline.recall)
    low=~helpful
    gain0=always.recall-baseline.recall
    gain1=variant.recall-baseline.recall
    def mean(series, mask):
        return float(series[mask].mean()) if mask.any() else np.nan
    denom=float(gain0[helpful].sum())
    return dict(queries=len(always),low_queries=int(low.sum()),helpful_queries=int(helpful.sum()),
        recall=float(variant.recall.mean()),ndc=float(variant.ndc.mean()),
        ndc_nn=float(variant.ndc_nn.mean()),
        recall_delta_pp=float((variant.recall-always.recall).mean()*100),
        ndc_delta=float((variant.ndc-always.ndc).mean()),
        zero_twohop_queries=int(variant.ndc_nn.eq(0).sum()),
        low_nn_always=mean(always.ndc_nn,low),low_nn_variant=mean(variant.ndc_nn,low),
        low_nn_saved=mean(always.ndc_nn-variant.ndc_nn,low),
        low_extra_ndc_always=mean(always.ndc-baseline.ndc,low),
        low_extra_ndc_variant=mean(variant.ndc-baseline.ndc,low),
        low_recall_delta_pp=mean((variant.recall-always.recall)*100,low),
        helpful_gain_always_pp=mean(gain0*100,helpful),
        helpful_gain_variant_pp=mean(gain1*100,helpful),
        helpful_net_gain_retention=float(gain1[helpful].sum()/denom) if denom>0 else np.nan,
        helpful_lower_than_always=int((variant.recall[helpful]<always.recall[helpful]).sum()),
        helpful_lower_than_baseline=int((variant.recall[helpful]<baseline.recall[helpful]).sum()),
        lower_than_always=int(variant.recall.lt(always.recall).sum()),
        higher_than_always=int(variant.recall.gt(always.recall).sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--indir',type=Path,required=True)
    ap.add_argument('--selection-dir',type=Path,help='Optional same-input selection sweep for always reproduction check')
    args=ap.parse_args()
    manifest=json.loads((args.indir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['suite']!='trigger': raise ValueError('Require an actual trigger-sweep manifest')
    always=load_run(args.indir/'trigger_always.csv')
    baseline=load_run(args.indir/'baseline.csv').loc[always.index]
    if args.selection_dir:
        previous=json.loads((args.selection_dir/'manifest.json').read_text(encoding='utf-8'))
        if previous['inputs']!=manifest['inputs'] or previous['controls_settings']!=manifest['controls_settings']:
            raise ValueError('Selection and trigger inputs or controls differ')
        selected=manifest['parameters']['selection']
        original=load_run(args.selection_dir/('select_'+selected+'.csv'))
        if not original.index.equals(always.index) or not np.array_equal(original.ndc,always.ndc) or not np.array_equal(original.recall,always.recall):
            raise ValueError('trigger_always does not reproduce the matching selection run; check local caps, mid cap and binary/index versions')
    rows=[]
    names=[case['name'] for case in manifest['cases'] if case['name'].startswith('trigger_')]
    for name in names:
        frame=load_run(args.indir/(name+'.csv'))
        if not frame.index.equals(always.index):raise ValueError('Query/ef mismatch: '+name)
        if not np.allclose(frame.paired_baseline_recall,baseline.recall) or not np.array_equal(frame.paired_baseline_ndc,baseline.ndc):
            raise ValueError('Paired baseline mismatch: '+name)
        zero=frame.ndc_nn.eq(0)
        if not np.array_equal(frame.recall[zero],baseline.recall[zero]) or not np.array_equal(frame.ndc[zero],baseline.ndc[zero]):
            raise ValueError('Zero-NN queries do not reproduce beam; inspect search decisions: '+name)
        for ef,g in frame.groupby(level='ef'):
            stats=group_metrics(g,always.loc[g.index],baseline.loc[g.index])
            stats.update(config=name,ef=int(ef),
                trigger_checks=float(g.gate_trigger_checks.mean()),
                trigger_fires=float(g.gate_trigger_fires.mean()),
                stall_fires=float(g.gate_stall_fires.mean()),
                frontier_fires=float(g.gate_frontier_fires.mean()))
            rows.append(stats)
    out=args.indir/'analysis';out.mkdir(exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(out/'trigger_group_summary.csv',index=False,na_rep='n.a.')
    report=['# Trigger group diagnostics','',
        'Groups are fixed from trigger_always versus same-ef beam: helpful means positive final recall gain; low means no positive gain. These GT-derived groups are offline diagnostics only.',
        'helpful_net_gain_retention = sum(trigger recall - beam recall) / sum(always recall - beam recall), on the fixed helpful group. It is signed aggregate net gain, not retained GT-ID overlap, and may exceed 1 or fall below 0.',
        'No-helpful-query groups have n.a. retention. Overall recall and total NDC remain the decision criteria; extra one-hop work can offset saved NN work.',
        'trigger_checks/fires count cur stages with eligible opportunities, not query counts. zero_twohop_queries counts queries with no fresh second-hop scores.',
        '',result[['config','ef','recall','ndc','ndc_nn','recall_delta_pp','ndc_delta',
            'low_queries','low_nn_saved','helpful_queries','helpful_net_gain_retention',
            'helpful_lower_than_baseline','zero_twohop_queries']].round(5).to_string(index=False)]
    (out/'trigger_group_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(result[['config','ef','recall_delta_pp','ndc_delta','low_nn_saved',
                  'helpful_net_gain_retention','helpful_lower_than_baseline']].round(5).to_string(index=False))
    print('Offline diagnostics saved; no online policy or search results were changed.')


if __name__=='__main__':main()
