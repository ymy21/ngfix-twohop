#!/usr/bin/env python3
"""Same-query original-clock comparison: cur/mid-only gates and larger-M beam."""
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


def grid(text):
    values=[int(x) for x in text.split(',')]
    if not values or min(values)<=0 or len(set(values))!=len(values):
        raise ValueError('Require distinct positive ef values')
    return sorted(values)


def profiles(gap,slack):
    # Same adjacency/NN rule throughout; only cur/mid gates change.
    off=sys.float_info.max
    return dict(mid_gap=(gap,0.,off),cur_only=(off,slack,off),cur_mid=(gap,slack,off),
                ratio_1=(off,slack,1.),ratio_2=(off,slack,2.),
                unfiltered=(off,0.,off))


def identity(path):
    st=path.stat()
    with path.open('rb') as f: header=f.read(56 if path.suffix=='.index' else 8)
    return dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,header_hex=header.hex())


def valid_graph_config(key,m,m0,mex):
    allowed=(16,32) if key=='index' else ((32,) if key=='m32_index' else (16,))
    return m in allowed and m0==2*m and mex==48


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bin','index','query','gt','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--m32-index',type=Path)
    p.add_argument('--ngfix-index',type=Path)
    p.add_argument('--k',type=int,default=100)
    p.add_argument('--efs',required=True)
    p.add_argument('--offset',type=int,default=1000)
    p.add_argument('--limit',type=int,default=1000)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--verify-limit',type=int,default=1000)
    p.add_argument('--gap',type=float,default=.0266)
    p.add_argument('--cur-slack',type=float,default=.0127)
    p.add_argument('--mid-cap',type=int,default=0)
    p.add_argument('--profiles',default='mid_gap,cur_mid,ratio_1')
    p.add_argument('--cpu',default='1')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();efs=grid(a.efs)
    if min(a.k,a.limit,a.rounds,a.verify_limit)<=0 or min(a.offset,a.mid_cap)<0 or min(efs)<a.k:
        p.error('Invalid population, cap or ef')
    if any(not math.isfinite(x) or x<0 for x in (a.gap,a.cur_slack)):
        p.error('Finite nonnegative thresholds required')
    names=a.profiles.split(',');available=profiles(a.gap,a.cur_slack)
    if not names or len(set(names))!=len(names) or any(x not in available for x in names) or 'cur_mid' not in names:
        p.error('Profiles must be distinct, valid, and include cur_mid as the reference case')
    if a.out.exists():p.error('Choose a new output directory')
    files={key:getattr(a,key).resolve() for key in ('index','query','gt','m32_index','ngfix_index') if getattr(a,key) is not None}
    inputs={} if a.dry_run else {key:identity(path) for key,path in files.items()}
    if not a.dry_run:
        if a.cpu!='none' and not shutil.which('taskset'):p.error('taskset missing')
        qn,qd=struct.unpack('<II',bytes.fromhex(inputs['query']['header_hex']))
        gn,gd=struct.unpack('<II',bytes.fromhex(inputs['gt']['header_hex']))
        if qn!=gn or gd<a.k or a.offset+a.limit>qn:p.error('Query/GT mismatch')
        populations=[]
        for key in ('index','m32_index','ngfix_index'):
            if key not in inputs:continue
            m,m0,mex,n,ep,dim,capacity=struct.unpack('<7Q',bytes.fromhex(inputs[key]['header_hex']))
            if dim!=qd or not 0<n<=capacity or ep>=n or m0!=2*m:p.error('Bad graph header: '+key)
            if not valid_graph_config(key,m,m0,mex):
                p.error('Unsupported graph config for '+key+': '+str((m,m0,mex)))
            populations.append(n)
        if len(set(populations))!=1:p.error('Graph populations differ')
        helptext=subprocess.run([str(a.bin.resolve()),'--help'],capture_output=True,text=True,check=True).stdout
        if 'Cur/mid gates full NN' not in helptext:p.error('Outdated benchmark')
    prefix=[] if a.cpu=='none' else ['taskset','-c',a.cpu]
    common=prefix+[str(a.bin.resolve()),'--query',str(files['query']),'--gt',str(files['gt']),
        '--metric','ip_float','--K',str(a.k),'--query-offset',str(a.offset),'--query-limit',str(a.limit),
        '--rounds',str(a.rounds),'--verify-limit',str(a.verify_limit),'--timing-protocol','original',
        '--warmup-passes','0','--seed','20261015','--beam-efs',a.efs,'--bound-efs',a.efs]
    cases=[]
    for name in names:
        gap,slack,ratio=available[name]
        cmd=common+['--index',str(files['index']),'--require-unrepaired','1','--mid-gap',str(gap),
            '--cur-margin',str(slack),'--detour-ratio',str(ratio),'--mid-cap',str(a.mid_cap)]
        cases.append(dict(name=name,beam_efs=efs,other_efs=efs,other_method='cur_mid_full',
                          mid_gap=gap,cur_slack=slack,detour_ratio=ratio,command=cmd))
    for key,label,pure in [('m32_index','hnsw_m32',1),('ngfix_index','ngfix_m16',0)]:
        if key in files:
            cmd=common+['--index',str(files[key]),'--require-unrepaired',str(pure),'--beam-only','1']
            cases.append(dict(name=label,beam_efs=efs,other_efs=[],other_method='cur_mid_full',command=cmd))
    for c in cases:c['command']+=['--out',str((a.out/c['name']/'timings.csv').resolve())]
    params={key:str(value) if isinstance(value,Path) else value for key,value in vars(a).items()}
    params.update(timing_protocol='original',min_batch_ms=300)
    manifest=dict(completed=False,suite='cur_mid_full',inputs=inputs,parameters=params,cases=cases,
        threshold_source='exploratory bins from earlier immediate probes; not an optimized or validated threshold',
        nn_selection='all fresh; no per-query/per-cur/per-mid NN budget',
        entry_policy='each saved graph entry; no forced common entry for M32')
    if not a.dry_run:manifest['binary_sha256']=hashlib.sha256(a.bin.read_bytes()).hexdigest()
    a.out.mkdir(parents=True)
    mp=a.out/'manifest.json'
    mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (a.out/'commands.txt').write_text('\n'.join(shlex.join(c['command']) for c in cases)+'\n',encoding='utf-8')
    if a.dry_run:print('Plan only:',a.out);return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for c in cases:
        folder=a.out/c['name'];folder.mkdir()
        print('Running '+c['name']+': '+str(folder),flush=True)
        with (folder/'benchmark.log').open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(c['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
            for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            if proc.wait():raise RuntimeError('Benchmark failed: '+c['name'])
    if {key:identity(path) for key,path in files.items()}!=inputs:raise RuntimeError('Input changed')
    manifest['completed']=True;mp.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_hnsw_cur_mid_full.py')),'--indir',str(a.out)],check=True)


if __name__=='__main__':main()
