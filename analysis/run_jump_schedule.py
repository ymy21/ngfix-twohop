#!/usr/bin/env python3
"""Run controlled selection/trigger sweeps; --dry-run writes reviewable commands only."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys


def csv_ints(text):
    values = [int(v) for v in text.split(',')]
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise ValueError('Require distinct positive integers')
    return sorted(values)


def build_cases(args):
    budget = ['--jump_mid_gap', str(args.gap), '--jump_max_nn_per_query', str(args.nn_cap)]
    scheduled = budget + ['--jump_schedule', 'after_onehop',
        '--jump_max_nn_per_mid', str(args.nn_per_mid), '--jump_max_nn_per_cur', str(args.nn_per_cur)]
    cases = [('baseline', 'baseline', []), ('original', 'boundtwohopcmp', []),
             ('legacy_budget', 'boundtwohopcmp', budget)]
    if args.suite == 'selection':
        cases += [('inline_adj', 'boundtwohopcmp', budget + ['--jump_max_mids_per_cur', str(args.mid_cap)]),
                  ('after_all', 'boundtwohopcmp', scheduled)]
        for mode in ('adjacency', 'gap', 'stratified'):
            cases.append(('select_' + mode, 'boundtwohopcmp', scheduled +
                ['--jump_mid_selection', mode, '--jump_max_mids_per_cur', str(args.mid_cap)]))
    else:
        common = scheduled + ['--jump_mid_selection', args.selection,
                              '--jump_max_mids_per_cur', str(args.mid_cap)]
        cases.append(('trigger_always', 'boundtwohopcmp', common))
        for width in csv_ints(args.windows):
            for mode in ('stall', 'stall_frontier'):
                cases.append((f'trigger_{mode}_w{width}', 'boundtwohopcmp', common +
                              ['--jump_trigger', mode, '--jump_stall_window', str(width)]))
    return cases


def input_identity(args):
    result = {}
    for label in ('query', 'gt', 'index'):
        path = getattr(args, label).resolve()
        with path.open('rb') as f:
            header = f.read(56 if label == 'index' else 8)
        result[label] = dict(path=str(path), bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns, header_hex=header.hex())
    qn, qd = struct.unpack('<II', bytes.fromhex(result['query']['header_hex']))
    gn, gd = struct.unpack('<II', bytes.fromhex(result['gt']['header_hex']))
    idx = struct.unpack('<7Q', bytes.fromhex(result['index']['header_hex']))
    if qn != gn or qd != idx[5] or gd < args.k or args.offset + args.limit > qn:
        raise ValueError('Query/GT/index shape, K or query range mismatch')
    for label, n, d in [('query', qn, qd), ('gt', gn, gd)]:
        if not n or not d or result[label]['bytes'] < 8 + 4*n*d:
            raise ValueError('Incomplete vector payload: ' + label)
    if not 0 < idx[3] <= idx[6] or idx[4] >= idx[3] or result['index']['bytes'] < 56 + idx[6]*(qd*4+1):
        raise ValueError('Invalid/incomplete index')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('suite', choices=['selection', 'trigger'])
    for name in ('bin', 'query', 'gt', 'index', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--offset', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--efs', default='100,200')
    ap.add_argument('--baseline-efs', default='100,150,200,300,400,500,600,800,1000')
    ap.add_argument('--gap', type=float, default=0.06)
    ap.add_argument('--mid-cap', type=int, default=2)
    ap.add_argument('--nn-cap', type=int, default=4000)
    ap.add_argument('--nn-per-mid', type=int, default=0)
    ap.add_argument('--nn-per-cur', type=int, default=0)
    ap.add_argument('--selection', choices=['adjacency', 'gap', 'stratified'], default='gap')
    ap.add_argument('--windows', default='4,8')
    ap.add_argument('--cpu', default='1', help='none disables taskset')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--controls-from', type=Path, help='Reuse controls from this runner with identical inputs')
    ap.add_argument('--include-dabs', action='store_true')
    ap.add_argument('--gammas', default='0,0.002,0.004,0.006,0.008,0.010,0.0125,0.015')
    args = ap.parse_args()
    efs, beam_efs = csv_ints(args.efs), csv_ints(args.baseline_efs)
    if args.k <= 0 or min(efs + beam_efs) < args.k or not set(efs).issubset(beam_efs):
        raise ValueError('Require ef>=K and baseline curve covering every experiment ef')
    if args.mid_cap < 2 or args.nn_cap <= 0 or args.offset < 0 or args.limit <= 0 or min(args.nn_per_mid, args.nn_per_cur) < 0:
        raise ValueError('Require mid-cap>=2, positive query budget/range, nonnegative local caps')
    if not math.isfinite(args.gap) or args.gap < 0:
        raise ValueError('Gap must be finite and nonnegative')
    gammas = [float(s) for s in args.gammas.split(',')]
    if not gammas or len(set(gammas)) != len(gammas) or any(not math.isfinite(g) or g < 0 for g in gammas):
        raise ValueError('Require distinct finite nonnegative DABS gammas')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a fresh output directory')
    identities = None if args.dry_run else input_identity(args)
    if not args.dry_run:
        if not args.bin.is_file() or not os.access(args.bin, os.X_OK):
            raise ValueError('Build search_hnsw_ngfix first')
        help_text = subprocess.run([str(args.bin.resolve()), '--help'], check=True, capture_output=True, text=True).stdout
        for option in ['--jump_schedule', '--jump_mid_selection', '--jump_trigger', '--jump_max_nn_per_mid', '--jump_max_nn_per_cur']:
            if option not in help_text: raise ValueError('Rebuild binary; missing ' + option)
    settings = dict(offset=args.offset, limit=args.limit, k=args.k, efs=efs, baseline_efs=beam_efs,
                    gap=args.gap, nn_cap=args.nn_cap)
    if args.controls_from:
        if args.dry_run: raise ValueError('Controls reuse requires real input validation')
        previous = json.loads((args.controls_from / 'manifest.json').read_text(encoding='utf-8'))
        if previous['inputs'] != identities or previous['controls_settings'] != settings:
            raise ValueError('Reusable controls differ in input identity or experiment settings')
        for name in ['baseline', 'original', 'legacy_budget']:
            if not (args.controls_from / (name + '.csv')).is_file(): raise ValueError('Missing reusable ' + name)
    prefix = []
    if args.cpu != 'none':
        if not args.dry_run and not shutil.which('taskset'): raise ValueError('taskset missing; use --cpu none')
        prefix = ['taskset', '-c', args.cpu]
    common = prefix + [str(args.bin.resolve()), '--test_query_path', str(args.query.resolve()),
        '--test_gt_path', str(args.gt.resolve()), '--index_path', str(args.index.resolve()),
        '--metric', 'ip_float', '--K', str(args.k), '--query_offset', str(args.offset),
        '--query_limit', str(args.limit), '--check_norms', '0', '--write_graph_audit', '0']
    records = []
    for name, mode, options in build_cases(args):
        command = common + ['--search_mode', mode, '--ef_values', ','.join(map(str, beam_efs if mode == 'baseline' else efs))] + options
        command += ['--result_path', str((args.out / (name + '.csv')).resolve())]
        records.append(dict(name=name, command=command,
                            reuse=bool(args.controls_from and name in ['baseline', 'original', 'legacy_budget'])))
    if args.include_dabs:
        records.append(dict(name='dabs', command=common + ['--search_mode', 'dabs', '--dabs_gammas', args.gammas,
            '--result_path', str((args.out / 'dabs' / 'dabs.csv').resolve())], reuse=False))
    manifest = dict(version=1, suite=args.suite, inputs=identities, controls_settings=settings,
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, cases=records)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (args.out / 'commands.txt').write_text('\n'.join(shlex.join(r['command']) for r in records) + '\n', encoding='utf-8')
    if args.dry_run:
        print(f'Plan only: {args.out.resolve()}; no ANN executed')
        return
    for rec in records:
        if rec['reuse']:
            shutil.copyfile(args.controls_from / (rec['name'] + '.csv'), args.out / (rec['name'] + '.csv'))
            continue
        if rec['name'] == 'dabs': (args.out / 'dabs').mkdir(exist_ok=True)
        print('Running ' + rec['name'], flush=True)
        with (args.out / (rec['name'] + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run(rec['command'], stdout=log, stderr=subprocess.STDOUT, check=True)
    for script in ['summarize_jump_gates.py', 'compare_jump_schedule.py']:
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)), '--indir', str(args.out)], check=True)
    print(f'Results: {args.out.resolve() / "analysis"}')


if __name__ == '__main__':
    main()
