#!/usr/bin/env python3
"""Unrepaired HNSW: capped-prefix ablations and independent clean timings."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys


def integers(text):
    values=[int(x) for x in text.split(',')]
    if not values or min(values)<=0 or len(set(values))!=len(values):
        raise ValueError('Require distinct positive ef values')
    return sorted(values)


def profiles(gap,margin):
    # Hold all budgets/NN selection fixed, changing only the geometric gates.
    return [('caps_only',1e100,0.0),('mid_gap',gap,0.0),('cur_mid',gap,margin)]


def input_identity(a):
    result={}
    for key in ('index','query','gt'):
        p=getattr(a,key).resolve()
        with p.open('rb') as f: header=f.read(56 if key=='index' else 8)
        result[key]=dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,header_hex=header.hex())
    qn,qd=struct.unpack('<II',bytes.fromhex(result['query']['header_hex']))
    gn,gd=struct.unpack('<II',bytes.fromhex(result['gt']['header_hex']))
    _,_,_,n,ep,dim,capacity=struct.unpack('<7Q',bytes.fromhex(result['index']['header_hex']))
    if qn!=gn or dim!=qd or gd<a.k or a.offset+a.limit>qn or not 0<n<=capacity or ep>=n:
        raise ValueError('Incompatible inputs or query range')
    if result['query']['bytes']<8+4*qn*qd or result['gt']['bytes']<8+4*gn*gd or result['index']['bytes']<56+capacity*(4*dim+1):
        raise ValueError('Truncated input')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('bin','unbounded-bin','index','query','gt','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--k',type=int,default=100)
    p.add_argument('--efs',default='100,110,120,130,140,150,160,180,200,250,300,400,500,800,1000,2000')
    p.add_argument('--unbounded-efs',default='100,200,400')
    p.add_argument('--offset',type=int,default=1000)
    p.add_argument('--limit',type=int,default=1000)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--verify-limit',type=int,default=1000)
    p.add_argument('--timing-protocol',choices=('original','batch'),default='original')
    p.add_argument('--min-batch-ms',type=int,default=300)
    p.add_argument('--gap',type=float,default=.0266)
    p.add_argument('--cur-margin',type=float,default=.0127)
    p.add_argument('--mid-cap',type=int,default=2)
    p.add_argument('--nn-limit',type=int,default=8)
    p.add_argument('--nn-cap',type=int,default=1024)
    p.add_argument('--cpu',default='1')
    p.add_argument('--skip-unbounded',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();efs=integers(a.efs);ub=integers(a.unbounded_efs)
    if min(a.k,a.limit,a.rounds,a.verify_limit,a.mid_cap,a.nn_limit,a.nn_cap)<=0 or a.offset<0 or a.min_batch_ms<10:
        p.error('Invalid query range, budgets or timing settings')
    if min(efs)<a.k or (not a.skip_unbounded and (min(ub)<a.k or not set(ub)<=set(efs))):
        p.error('Require ef>=K and unbounded ef subset of main grid')
    if any(not math.isfinite(x) or x<0 for x in (a.gap,a.cur_margin)):
        p.error('Distance thresholds must be finite and nonnegative')
    if a.out.exists(): p.error('Use a new output directory; previous results are preserved')
    inputs=None if a.dry_run else input_identity(a)
    if not a.dry_run:
        if a.cpu!='none' and not shutil.which('taskset'): p.error('taskset missing')
        binaries=[(a.bin,'Selective prefix after one-hop')]
        if not a.skip_unbounded: binaries.append((a.unbounded_bin,'Unbounded default boundtwohop'))
        for binary,marker in binaries:
            helptext=subprocess.run([str(binary.resolve()),'--help'],check=True,capture_output=True,text=True).stdout
            if marker not in helptext: p.error('Wrong/outdated binary: '+str(binary))
    prefix=[] if a.cpu=='none' else ['taskset','-c',a.cpu]
    common=['--index',str(a.index.resolve()),'--query',str(a.query.resolve()),'--gt',str(a.gt.resolve()),
        '--metric','ip_float','--K',str(a.k),'--query-offset',str(a.offset),'--query-limit',str(a.limit),
        '--rounds',str(a.rounds),'--verify-limit',str(a.verify_limit),'--timing-protocol',a.timing_protocol,
        '--warmup-passes','0' if a.timing_protocol=='original' else '1','--min-batch-ms',str(a.min_batch_ms),
        '--seed','20261015','--require-unrepaired','1','--beam-efs',a.efs]
    cases=[]
    if not a.skip_unbounded:
        cases.append(dict(name='control',beam_efs=efs,other_efs=ub,other_method='bound_unbounded',
            command=prefix+[str(a.unbounded_bin.resolve())]+common+['--bound-efs',a.unbounded_efs]))
    for name,gap,margin in profiles(a.gap,a.cur_margin):
        cases.append(dict(name=name,beam_efs=efs,other_efs=efs,other_method='selective_prefix',gap=gap,cur_margin=margin,
            command=prefix+[str(a.bin.resolve())]+common+['--bound-efs',a.efs,'--mid-gap',str(gap),
                '--cur-margin',str(margin),'--mid-cap',str(a.mid_cap),'--nn-limit',str(a.nn_limit),'--nn-cap',str(a.nn_cap)]))
    for c in cases:
        c['command']+=['--out',str((a.out/c['name']/'timings.csv').resolve())]
    m=dict(completed=False,suite='hnsw_selective_prefix',inputs=inputs,cases=cases,
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        threshold_source='exploratory HNSW K10 ef100 bins: gap Q1=.0266, cur slack median=.0127; not fitted optimum',
        counters='outside timing; actual prefix kernel scans, not reference full-pool census')
    if not a.dry_run:
        m['binary_sha256']={str(b):hashlib.sha256(b.read_bytes()).hexdigest() for b,_ in binaries}
    a.out.mkdir(parents=True)
    manifest=a.out/'manifest.json'
    manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    (a.out/'commands.txt').write_text('\n'.join(shlex.join(c['command']) for c in cases)+'\n',encoding='utf-8')
    if a.dry_run: print('Plan only:',a.out);return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for c in cases:
        folder=a.out/c['name'];folder.mkdir()
        print('Running '+c['name']+': '+str(folder),flush=True)
        with (folder/'benchmark.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(c['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
            for line in proc.stdout: print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait(): raise RuntimeError('Benchmark failed: '+c['name'])
    if input_identity(a)!=inputs: raise RuntimeError('Inputs changed during experiment')
    m['completed']=True;manifest.write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_hnsw_selective_prefix.py')),'--indir',str(a.out)],check=True)


if __name__=='__main__': main()
