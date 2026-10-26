#!/usr/bin/env python3
"""One process, one index, independent beam/cheap32 timing points."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import sys


def integers(text):
    values=[int(v) for v in text.split(',')]
    if not values or min(values)<=0 or len(set(values))!=len(values): raise ValueError('Distinct positive ef values required')
    return sorted(values)


def identity(a):
    identities={}
    for name in ['query','gt','index']:
        file=getattr(a,name).resolve()
        with file.open('rb') as f: header=f.read(56 if name=='index' else 8)
        identities[name]=dict(path=str(file),bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns,header_hex=header.hex())
    qn,qd=struct.unpack('<II',bytes.fromhex(identities['query']['header_hex']))
    gn,gd=struct.unpack('<II',bytes.fromhex(identities['gt']['header_hex']))
    index=struct.unpack('<7Q',bytes.fromhex(identities['index']['header_hex']))
    if qn!=gn or qd!=index[5] or qd<32 or gd<a.k or a.offset+a.limit>qn:
        raise ValueError('Query/GT/index dimensions or query range mismatch')
    for name,n,d in [('query',qn,qd),('gt',gn,gd)]:
        if identities[name]['bytes']<8+4*n*d: raise ValueError('Truncated '+name)
    if not 0<index[3]<=index[6] or index[4]>=index[3] or identities['index']['bytes']<56+index[6]*(4*qd+1):
        raise ValueError('Invalid/truncated index')
    return identities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bin','query','gt','index','out']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=10000)
    p.add_argument('--k',type=int,default=100)
    p.add_argument('--beam-efs',default='100,120,140,160,180,200,220,240,260,280,300,325,350,375,400,450,500,550,600,700,800,900,1000')
    p.add_argument('--cheap-efs',default='100,110,120,130,140,150,175,200,225,250,275,300,325,350,375,400,450,500,550,600')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--warmup-passes',type=int,default=1)
    p.add_argument('--verify-limit',type=int,default=200)
    p.add_argument('--min-batch-ms',type=int,default=300);p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--cpu',default='1');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--targets',default='0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.992,0.995')
    a=p.parse_args();beam=integers(a.beam_efs);cheap=integers(a.cheap_efs)
    if min(beam+cheap)<a.k or min(a.k,a.limit,a.rounds,a.verify_limit)<=0 or min(a.offset,a.warmup_passes,a.seed)<0 or a.min_batch_ms<10:
        p.error('Invalid ef/K/query/timing settings')
    targets=[float(v) for v in a.targets.split(',')]
    if not targets or any(not math.isfinite(v) or not 0<=v<=1 for v in targets): p.error('Invalid targets')
    if a.out.exists() and any(a.out.iterdir()): p.error('Use a fresh output directory')
    inputs=None if a.dry_run else identity(a)
    if not a.dry_run:
        if importlib.util.find_spec('matplotlib') is None:
            p.error('Automatic plotting requires matplotlib: python3 -m pip install matplotlib')
        if not a.bin.is_file() or not os.access(a.bin,os.X_OK): p.error('Build bench_hnsw_ngfix_clean first')
        helptext=subprocess.run([str(a.bin.resolve()),'--help'],check=True,capture_output=True,text=True).stdout
        if 'Fixed cheap32/NN8' not in helptext: p.error('Wrong benchmark binary')
    prefix=[] if a.cpu=='none' else ['taskset','-c',a.cpu]
    command=prefix+[str(a.bin.resolve()),'--index',str(a.index.resolve()),'--query',str(a.query.resolve()),
        '--gt',str(a.gt.resolve()),'--out',str((a.out/'timings.csv').resolve()),'--metric','ip_float',
        '--K',str(a.k),'--query-offset',str(a.offset),'--query-limit',str(a.limit),
        '--beam-efs',','.join(map(str,beam)),'--cheap-efs',','.join(map(str,cheap)),
        '--rounds',str(a.rounds),'--warmup-passes',str(a.warmup_passes),
        '--min-batch-ms',str(a.min_batch_ms),'--seed',str(a.seed),'--verify-limit',str(a.verify_limit)]
    parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    parameters['beam_efs']=beam;parameters['cheap_efs']=cheap
    m=dict(version=1,suite='clean_qps',completed=False,inputs=inputs,parameters=parameters,command=command,
        profile=dict(schedule='after_onehop',mid_selection='gap',gap=.06,mid_cap=2,trigger='always',
                     nn_cap=4000,nn_limit=8,cheap_dims=32,first_edge='all',second_edge='all',policy='off'),
        platform=platform.platform(),single_worker=True)
    if not a.dry_run: m['binary_sha256']=hashlib.sha256(a.bin.read_bytes()).hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=a.out/'manifest.json'
    manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    (a.out/'commands.txt').write_text(shlex.join(command)+'\n',encoding='utf-8')
    if a.dry_run: print('Plan only:',a.out); return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    print('Reference verification, warmup and randomized timing rounds run in one process.',flush=True)
    with (a.out/'benchmark.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        if process.wait(): raise RuntimeError('Benchmark failed; see benchmark.log. No completed comparison.')
    m['completed']=True;manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_clean_qps.py')),
                    '--indir',str(a.out),'--targets',a.targets],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('plot_clean_qps.py')),
                    '--indir',str(a.out)],check=True)
    print('Results:',a.out/'analysis')

if __name__=='__main__': main()
