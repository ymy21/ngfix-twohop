#!/usr/bin/env python3
"""Paired mechanism comparisons; optionally include same-query DABS curve."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from summarize_jump_gates import load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--indir', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.indir / 'manifest.json').read_text(encoding='utf-8'))
    names = [r['name'] for r in manifest['cases'] if r['name'] != 'dabs']
    runs = {name: load_run(args.indir / (name + '.csv')) for name in names}
    reference = 'select_adjacency' if manifest['suite'] == 'selection' else 'trigger_always'
    ref = runs[reference]
    expected = list(range(manifest['controls_settings']['offset'],
                          manifest['controls_settings']['offset'] + manifest['controls_settings']['limit']))
    for name, frame in runs.items():
        if frame is None: raise ValueError('No QUERY rows in ' + name)
        for _, group in frame.groupby(level='ef'):
            if group.index.get_level_values('query_id').sort_values().tolist() != expected:
                raise ValueError('Query IDs mismatch: ' + name)
        if name not in ['baseline'] and not frame.index.equals(ref.index):
            raise ValueError('Different ef/query pairs: ' + name)
        if name not in ['baseline', 'original', 'legacy_budget']:
            nn = frame.ndc_nn
            settings = manifest['parameters']
            if (nn > settings['nn_cap']).any(): raise ValueError('Query NDC cap exceeded: ' + name)
            for flag, counter in [('nn_per_mid', 'twohop_mid_scans'), ('nn_per_cur', 'hops')]:
                cap = settings[flag]
                if cap and (nn > cap*frame[counter]).any(): raise ValueError('Local NDC cap exceeded: ' + name)
            type_sum = frame.gate_probed_base_mids + frame.gate_probed_ngfix_mids + frame.gate_probed_rfix_mids
            if not np.array_equal(type_sum.to_numpy(), frame.twohop_mid_scans.to_numpy()):
                raise ValueError('Probe provenance accounting mismatch: ' + name)
    paired, burden = [], []
    for name, frame in runs.items():
        if name == 'baseline': continue
        for ef, group in frame.groupby(level='ef'):
            control = ref.loc[group.index]
            dr, dn = group.recall-control.recall, group.ndc-control.ndc
            paired.append(dict(config=name, reference=reference, ef=ef, queries=len(group),
                recall_delta=dr.mean(), recall_delta_se=dr.std(ddof=1)/np.sqrt(len(group)),
                ndc_delta=dn.mean(), ndc_delta_se=dn.std(ddof=1)/np.sqrt(len(group)),
                cost_without_recall_gain_queries=int(((dn>0)&(dr<=0)).sum()),
                lower_recall_queries=int((dr<0).sum())))
            for key in group.index:
                burden.append(dict(config=name, reference=reference, ef=ef, query_id=key[1],
                                   recall_delta=dr.loc[key], ndc_delta=dn.loc[key]))
    out = args.indir / 'analysis'
    out.mkdir(exist_ok=True)
    pd.DataFrame(paired).to_csv(out / 'mechanism_comparison.csv', index=False)
    pd.DataFrame(burden).to_csv(out / 'query_deltas.csv', index=False)
    points = pd.read_csv(out / 'summary.csv')[['config','ef','recall','ndc']].copy()
    dabs = args.indir / 'dabs' / 'dabs.csv'
    if dabs.exists():
        frame = pd.read_csv(dabs)
        frame = frame[frame.row_type.eq('QUERY')]
        for gamma, group in frame.groupby('dabs_gamma'):
            if group.query_id.sort_values().tolist() != expected: raise ValueError('DABS query IDs mismatch')
            if not np.allclose(group.ndc, group.ndc_entry+group.ndc_mid+group.ndc_nn):
                raise ValueError('DABS NDC accounting mismatch')
            points.loc[len(points)] = [f'dabs_gamma{gamma:g}', 0, group.recall.mean(), group.ndc.mean()]
    points.to_csv(out / 'combined_points.csv', index=False)
    rows = []
    for target in [.98,.99,.995]:
        for method in ['beam','dabs','new']:
            mask = points.config.eq('baseline') if method=='beam' else points.config.str.startswith('dabs_gamma') if method=='dabs' else points.config.str.startswith(('select_', 'trigger_'))
            candidates = points[mask & points.recall.ge(target)]
            best = candidates.sort_values('ndc').head(1)
            rows.append(dict(target_recall=target, method=method, reached=not best.empty,
                config='' if best.empty else best.iloc[0].config,
                ef=np.nan if best.empty else best.iloc[0].ef,
                achieved_recall=np.nan if best.empty else best.iloc[0].recall,
                ndc=np.nan if best.empty else best.iloc[0].ndc))
    pd.DataFrame(rows).to_csv(out / 'combined_targets.csv', index=False)
    print('Paired comparisons and query deltas saved; query labels are offline diagnostics only.')


if __name__ == '__main__':
    main()
