#!/usr/bin/env python3
"""Stage 1: one graph, clean beam vs unbounded default boundtwohop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from run_clean_qps import identity, integers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bin', 'query', 'gt', 'index', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--efs', default='100,120,150,180,200,250,300,400,500,600,800,1000,1200,1600,2400')
    p.add_argument('--k', type=int, default=100)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=10000)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--verify-limit', type=int, default=200)
    p.add_argument('--warmup-passes', type=int)
    p.add_argument('--timing-protocol', choices=('batch','original'), default='batch')
    p.add_argument('--min-batch-ms', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261013)
    p.add_argument('--cpu', default='1')
    p.add_argument('--allow-repaired', action='store_true', help='Only for separately labeled NGFix reference runs')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.warmup_passes is None:
        a.warmup_passes=0 if a.timing_protocol=='original' else 1
    if a.timing_protocol=='original' and a.warmup_passes:
        p.error('Original protocol requires --warmup-passes 0')
    efs = integers(a.efs)
    if min(efs) < a.k or min(a.k,a.limit,a.rounds,a.verify_limit) <= 0 or min(a.offset,a.warmup_passes,a.seed) < 0 or a.min_batch_ms < 10:
        p.error('Invalid K/ef/query/timing settings')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a fresh output directory; previous results are preserved')
    inputs = None if a.dry_run else identity(a)
    if not a.dry_run:
        helptext = subprocess.run([str(a.bin.resolve()), '--help'], check=True, capture_output=True, text=True).stdout
        if 'Unbounded default boundtwohop' not in helptext:
            p.error('Build bench_hnsw_unbounded_clean first; wrong benchmark binary')
        for key in ('query','gt'):
            inputs[key]['sha256'] = hashlib.sha256(getattr(a,key).read_bytes()).hexdigest()
    prefix = [] if a.cpu == 'none' else ['taskset', '-c', a.cpu]
    cmd = prefix + [str(a.bin.resolve()), '--index', str(a.index.resolve()), '--query', str(a.query.resolve()),
                   '--gt', str(a.gt.resolve()), '--out', str((a.out/'timings.csv').resolve()), '--metric', 'ip_float',
                   '--K', str(a.k), '--query-offset', str(a.offset), '--query-limit', str(a.limit),
                   '--beam-efs', a.efs, '--bound-efs', a.efs, '--rounds', str(a.rounds),
                   '--warmup-passes', str(a.warmup_passes), '--min-batch-ms', str(a.min_batch_ms),
                   '--seed', str(a.seed), '--verify-limit', str(a.verify_limit),
                   '--timing-protocol', a.timing_protocol,
                   '--require-unrepaired', '0' if a.allow_repaired else '1']
    m = dict(suite='hnsw_stage1', completed=False, inputs=inputs,
             graph='repaired_reference' if a.allow_repaired else 'unrepaired_bottom',
             profile=dict(schedule='immediate',trigger='always',nn_selection='full',
                          gates='off',budgets='unlimited',edge_filters='all',policy='off'),
             parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
             command=cmd, platform=platform.platform())
    m['parameters']['efs'] = efs
    if not a.dry_run:
        m['binary_sha256'] = hashlib.sha256(a.bin.read_bytes()).hexdigest()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out/'manifest.json'
    manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    (a.out/'commands.txt').write_text(shlex.join(cmd)+'\n',encoding='utf-8')
    if a.dry_run:
        print('Plan only:', a.out)
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (a.out/'benchmark.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line); log.flush()
        if process.wait():
            raise RuntimeError('Benchmark failed; manifest remains incomplete')
    # Verify input identity after the sweep too; QPS is not useful for changing files.
    after = identity(a)
    for key in ('index','query','gt'):
        for field in ('bytes','mtime_ns','header_hex'):
            if after[key][field] != inputs[key][field]:
                raise RuntimeError('Input changed during sweep: '+key)
    m['completed'] = True
    manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_hnsw_stage1.py')),
                    '--indir',str(a.out)],check=True)


if __name__ == '__main__':
    main()
