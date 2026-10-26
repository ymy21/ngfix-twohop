#!/usr/bin/env python3
"""NN diagnosis (shadow ranks) or actual online ablation on identical query IDs."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from run_jump_schedule import csv_ints, input_identity


def build_cases(a):
    common = ['--jump_schedule', 'after_onehop', '--jump_mid_selection', 'gap',
              '--jump_mid_gap', str(a.gap), '--jump_max_mids_per_cur', str(a.mid_cap),
              '--jump_trigger', 'always', '--jump_max_nn_per_query', str(a.nn_cap),
              '--jump_max_nn_per_mid', '0', '--jump_max_nn_per_cur', '0',
              '--jump_nn_limit', str(a.nn_limit)]
    if a.suite == 'diagnostic':
        return [(f'nn_full_ef{ef}', 'boundtwohopcmp', str(ef), common +
                 ['--jump_nn_selection', 'full', '--jump_nn_seed', str(a.seeds[0]),
                  '--jump_nn_cheap_dims', str(a.cheap_dims[0]), '--nn_trace', '1',
                  '--event_trace_ef', str(ef), '--event_trace_query_limit', str(a.limit),
                  '--event_trace_compact', '1']) for ef in csv_ints(a.efs)]
    cases = [('baseline', 'baseline', a.baseline_efs, []),
             ('nn_full', 'boundtwohopcmp', a.efs, common + ['--jump_nn_selection', 'full']),
             (f'nn_prefix{a.nn_limit}', 'boundtwohopcmp', a.efs, common + ['--jump_nn_selection', 'prefix'])]
    cases += [(f'nn_random{a.nn_limit}_s{s}', 'boundtwohopcmp', a.efs, common +
               ['--jump_nn_selection', 'random', '--jump_nn_seed', str(s)]) for s in a.seeds]
    cases += [(f'nn_cheap{a.nn_limit}_d{d}', 'boundtwohopcmp', a.efs, common +
               ['--jump_nn_selection', 'cheap', '--jump_nn_cheap_dims', str(d)]) for d in a.cheap_dims]
    if a.include_dabs:
        cases.append(('dabs', 'dabs', a.efs, ['--dabs_gammas', a.gammas]))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['diagnostic', 'online'], default='diagnostic')
    for name in ('bin', 'query', 'gt', 'index', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--offset', type=int, default=1000)
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--efs', default='100,200')
    p.add_argument('--baseline-efs', default='100,110,120,130,140,150,200,250,300,400,500,600,800,1000')
    p.add_argument('--gap', type=float, default=.06)
    p.add_argument('--mid-cap', type=int, default=2)
    p.add_argument('--nn-cap', type=int, default=4000)
    p.add_argument('--nn-limit', type=int, default=8)
    p.add_argument('--cheap-dims', default='16')
    p.add_argument('--seeds', default='17')
    p.add_argument('--cpu', default='1')
    p.add_argument('--include-dabs', action='store_true')
    p.add_argument('--gammas', default='0,0.0005,0.001,0.0015,0.002,0.004,0.006,0.008,0.010,0.0125,0.015')
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    a.cheap_dims=csv_ints(a.cheap_dims); a.seeds=csv_ints(a.seeds)
    if a.suite=='diagnostic' and (len(a.cheap_dims)!=1 or len(a.seeds)!=1):
        p.error('One cheap dimension count and one random seed per shadow diagnosis; use separate output directories')
    efs=csv_ints(a.efs); beam=csv_ints(a.baseline_efs)
    if min(efs+beam)<a.k or not set(efs)<=set(beam) or a.k<=0:
        p.error('ef must cover K; baseline must cover each ablation ef')
    if a.offset<0 or a.limit<=0 or a.nn_cap<0 or min(a.mid_cap,a.nn_limit)<=0:
        p.error('Invalid query range or budget (NN_CAP=0 means unlimited)')
    if not math.isfinite(a.gap) or a.gap<0: p.error('Gap must be finite and nonnegative')
    gs=[float(v) for v in a.gammas.split(',')]
    if not gs or any(not math.isfinite(g) or g<0 for g in gs) or len(set(gs))!=len(gs):
        p.error('Invalid DABS gamma list')
    if a.out.exists() and any(a.out.iterdir()): p.error('Use a fresh output directory')
    identities=None if a.dry_run else input_identity(a)
    if not a.dry_run:
        if not a.bin.is_file() or not os.access(a.bin,os.X_OK): p.error('Build search_hnsw_ngfix first')
        help_text=subprocess.run([str(a.bin.resolve()),'--help'],check=True,capture_output=True,text=True).stdout
        if '--jump_nn_selection' not in help_text or '--nn_trace' not in help_text:
            p.error('Binary has no NN selection support; rebuild')
        dimension=int.from_bytes(bytes.fromhex(identities['query']['header_hex'])[4:8],'little')
        if max(a.cheap_dims)>dimension: p.error('Cheap dimensions exceed vector dimension')
    else: dimension=None
    prefix=[] if a.cpu=='none' else ['taskset','-c',a.cpu]
    command=prefix+[str(a.bin.resolve()),'--test_query_path',str(a.query.resolve()),
        '--test_gt_path',str(a.gt.resolve()),'--index_path',str(a.index.resolve()),
        '--metric','ip_float','--K',str(a.k),'--query_offset',str(a.offset),
        '--query_limit',str(a.limit),'--check_norms','0','--write_graph_audit','0']
    records=[]
    for name,mode,ef,options in build_cases(a):
        path=a.out/(('dabs/dabs' if name=='dabs' else name)+'.csv')
        records.append(dict(name=name,command=command+['--search_mode',mode,'--ef_values',ef]+options+
                            ['--result_path',str(path.resolve())]))
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(version=1,suite='nn_'+a.suite,inputs=identities,dimension=dimension,
        controls_settings=dict(offset=a.offset,limit=a.limit,k=a.k,efs=efs),
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},cases=records)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (a.out/'commands.txt').write_text('\n'.join(shlex.join(r['command']) for r in records)+'\n',encoding='utf-8')
    if a.dry_run:
        print('Plan only:',a.out); return
    for record in records:
        if record['name']=='dabs': (a.out/'dabs').mkdir()
        print('Running '+record['name'],flush=True)
        with (a.out/(record['name']+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(record['command'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_jump_nn_selection.py')),
                    '--indir',str(a.out)],check=True)
    print('Results:',a.out/'analysis')

if __name__=='__main__': main()
