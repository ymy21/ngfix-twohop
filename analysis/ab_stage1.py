#!/usr/bin/env python3
"""Prepare train-only A/B experiments, build labels, run FULL-NN benchmarks.

NumPy is required for prepare; optional faiss provides exact IndexFlatIP search.
All outputs use fresh paths. Existing graph/query/GT files are read-only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time


def read_header(path, item_size=4, allow_tail=False):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f'Incomplete header: {path}')
    n, d = struct.unpack('<II', raw)
    expected = 8 + n * d * item_size
    size = path.stat().st_size
    if not n or not d or size < expected or (not allow_tail and size != expected):
        raise ValueError(f'Invalid header/payload: {path}; shape={n}x{d}, bytes={size}, expected={expected}')
    return n, d, size - expected


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def discover_gt(base, query, expected_rows, top_l):
    candidates = []
    for folder in sorted({Path(base).resolve().parent, Path(query).resolve().parent}):
        for pattern in ('*gt*train*.bin', '*gt*train*.ibin', '*train*gt*.bin', '*train*gt*.ibin'):
            for p in folder.glob(pattern):
                try:
                    n, d, _ = read_header(p, allow_tail=True)
                    if n == expected_rows and d >= top_l:
                        candidates.append(p.resolve())
                except (ValueError, OSError):
                    pass
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        raise ValueError('Multiple plausible training GT files; specify --train-gt:\n' + '\n'.join(map(str, candidates)))
    return candidates[0] if candidates else None


def exact_topk(base, queries, k, block=32768, backend='numpy'):
    """Exact exhaustive IP: float64 NumPy blocks or float32 FAISS IndexFlatIP.

    NumPy ties use ascending base ID. FAISS equal-score tie order is unspecified.
    No ANN-generated list is silently treated as exact ground truth.
    """
    import numpy as np
    if not 0 < k <= len(base) or not block:
        raise ValueError('Invalid exact-GT k/block')
    if backend == 'faiss':
        import faiss
        index = faiss.IndexFlatIP(base.shape[1])
        for start in range(0, len(base), block):
            chunk = np.asarray(base[start:start + block], dtype=np.float32, order='C')
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite base values')
            index.add(chunk)
        _, ids = index.search(np.asarray(queries, dtype=np.float32, order='C'), k)
        return ids.astype('<i4')
    q = np.asarray(queries, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite query values')
    best_ids = np.empty((len(q), 0), dtype=np.int64)
    best_scores = np.empty((len(q), 0), dtype=np.float64)
    for start in range(0, len(base), block):
        vectors = np.asarray(base[start:start + block], dtype=np.float64)
        if not np.isfinite(vectors).all():
            raise ValueError('Nonfinite base values')
        scores = q @ vectors.T
        ids = np.broadcast_to(np.arange(start, start + len(vectors)), scores.shape)
        scores = np.concatenate((best_scores, scores), axis=1)
        ids = np.concatenate((best_ids, ids), axis=1)
        take = min(k, scores.shape[1])
        partial = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        out_ids = np.empty((len(q), take), dtype=np.int64)
        out_scores = np.empty((len(q), take), dtype=np.float64)
        for row in range(len(q)):
            threshold = scores[row, partial[row]].min()
            tied_pool = np.flatnonzero(scores[row] >= threshold)
            order = np.lexsort((ids[row, tied_pool], -scores[row, tied_pool]))[:take]
            selected = tied_pool[order]
            out_ids[row] = ids[row, selected]
            out_scores[row] = scores[row, selected]
        best_ids, best_scores = out_ids, out_scores
    return best_ids.astype('<i4')


def resolve_backend(name):
    if name == 'auto':
        try:
            import faiss  # noqa: F401
            return 'faiss'
        except ImportError:
            return 'numpy'
    if name == 'faiss':
        import faiss  # noqa: F401
    return name


def validate_gt_rows(rows, nbase):
    import numpy as np
    if (rows < 0).any() or (rows >= nbase).any():
        raise ValueError('GT ID outside base range')
    sorted_ids = np.sort(rows, axis=1)
    if rows.shape[1] > 1 and (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
        raise ValueError('Duplicate GT IDs within a row')


def audit_gt(args):
    """Read-only check of candidate GTs; one exhaustive scan for shared sample IDs."""
    import numpy as np
    nbase, dim, _ = read_header(args.base)
    nq, qdim, _ = read_header(args.query)
    if dim != qdim or not 0 < args.top_l <= nbase:
        raise ValueError('Base/query dimension or top-L mismatch')
    if args.out.exists():
        raise ValueError('Choose a fresh audit report path')
    ids = np.sort(np.random.default_rng(args.seed).choice(nq, min(args.samples, nq), replace=False))
    report = {'base': identity(args.base), 'query': identity(args.query), 'metric': 'inner_product',
              'base_count': nbase, 'query_count': nq, 'dimension': dim, 'sample_ids': ids.tolist(),
              'top_l_requested': args.top_l, 'candidates': [],
              'scope': 'Sampled rows only; passing is not a proof for every GT row.'}
    candidates = []
    for path in args.gt:
        record = {'path': str(path.resolve()), 'status': 'invalid'}
        report['candidates'].append(record)
        try:
            n, width, tail = read_header(path, allow_tail=True)
            record.update(rows=n, width=width, trailing_bytes=tail)
            if n != nq:
                raise ValueError('GT/query row counts differ')
            if tail not in (0, n * width * 4):
                raise ValueError('Unsupported trailing layout (expected IDs only or IDs followed by float distances)')
            gt = np.memmap(path, dtype='<i4', mode='r', offset=8, shape=(n, width))
            rows = np.array(gt[ids], copy=True); del gt
            validate_gt_rows(rows, nbase)
            record['sampled_id_range_and_uniqueness'] = 'passed'
            candidates.append((record, rows, min(args.top_l, width)))
        except (OSError, ValueError) as exc:
            record['error'] = str(exc)
    if candidates:
        base = np.memmap(args.base, dtype='<f4', mode='r', offset=8, shape=(nbase, dim))
        queries = np.memmap(args.query, dtype='<f4', mode='r', offset=8, shape=(nq, dim))
        q = np.asarray(queries[ids], dtype=np.float64)
        print(f'Exhaustive IP audit: {len(ids)} queries x {nbase} base vectors; float64 blocked search.', flush=True)
        expected = exact_topk(base, q, max(c[2] for c in candidates), args.base_block)
        for record, rows, k in candidates:
            levels = sorted({min(100, k), k})
            record['prefix_checks'] = {}
            passed = True
            for level in levels:
                equal = np.all(np.sort(rows[:, :level], axis=1) == np.sort(expected[:, :level], axis=1), axis=1)
                overlaps = [len(set(a).intersection(b)) / level for a, b in zip(rows[:, :level], expected[:, :level])]
                record['prefix_checks'][str(level)] = {'exact_set_matches': int(equal.sum()),
                    'sample_count': len(ids), 'mean_overlap': float(np.mean(overlaps)),
                    'failed_query_ids': ids[~equal].tolist()}
                passed = passed and bool(equal.all())
            record['status'] = ('passed' if k == args.top_l else 'matched_but_too_narrow') if passed else 'mismatch'
            record['usable_for_requested_l'] = record['status'] == 'passed'
        del queries, base
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(args.out, report)
    for record in report['candidates']:
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print('Audit report:', args.out, flush=True)
    if not any(r.get('usable_for_requested_l') for r in report['candidates']):
        raise ValueError('No candidate passed at requested L. Inspect report: mapping, metric, width, and boundary ties may matter.')


def audit_index_base(index, base_path, samples=32):
    """Check the legacy raw-float index uses the same base ID order."""
    import numpy as np
    n, dim, _ = read_header(base_path)
    with Path(index).open('rb') as stream:
        header = stream.read(56)
        if len(header) != 56:
            raise ValueError('Incomplete raw-float index header')
        _, _, _, gn, entry, gd, capacity = struct.unpack('<7Q', header)
        if gn != n or gd != dim or capacity < gn or entry >= gn:
            raise ValueError('Index and base metadata mismatch; requires original unrepaired raw-float HNSW bottom format')
        if Path(index).stat().st_size < 56 + capacity * (dim * 4 + 1) + n * 4:
            raise ValueError('Incomplete raw-float index')
        ids = np.unique(np.concatenate(([0, n - 1], np.random.default_rng(17).integers(0, n, min(samples, n)))))
        base = np.memmap(base_path, dtype='<f4', mode='r', offset=8, shape=(n, dim))
        for node in ids:
            stream.seek(56 + int(node) * (dim * 4 + 1) + 1)  # deletion byte precedes vector
            vector = np.frombuffer(stream.read(dim * 4), dtype='<f4')
            if not np.array_equal(vector, base[node]):
                raise ValueError(f'Index/base vector or ID-order mismatch at node {node}')
        del base
    return {'status': 'passed', 'sampled_nodes': ids.tolist(), 'base_count': n, 'dimension': dim}


def prepare(args):
    import numpy as np
    nbase, dim, _ = read_header(args.base)
    nq, qdim, _ = read_header(args.train_query)
    if dim != qdim or dim != 512 or args.top_l < 100 or args.top_l > nbase:
        raise ValueError('This WebVid experiment requires matching 512D vectors and 100<=L<=N')
    if args.out.exists():
        raise ValueError('Choose a fresh prepare output directory')
    base = np.memmap(args.base, dtype='<f4', mode='r', offset=8, shape=(nbase, dim))
    queries = np.memmap(args.train_query, dtype='<f4', mode='r', offset=8, shape=(nq, dim))
    source_gt = (discover_gt(args.base, args.train_query, nq, args.top_l)
                 if args.train_gt == 'auto' else (None if args.train_gt == 'none' else Path(args.train_gt).resolve()))
    gt = None
    if source_gt:
        gn, gd, tail = read_header(source_gt, allow_tail=True)
        if gn != nq or gd < args.top_l:
            raise ValueError('Training GT row count/width mismatch')
        print(f'Using training GT: {source_gt}; width={gd}, trailing bytes={tail}', flush=True)
        gt = np.memmap(source_gt, dtype='<i4', mode='r', offset=8, shape=(gn, gd))
    else:
        print('No usable training GT selected: exact GT will be computed ONLY for sampled train/holdout rows.', flush=True)
    backend = resolve_backend(args.backend)
    args.out.mkdir(parents=True)
    cmd = [sys.executable, str(Path(__file__).with_name('prepare_ab_query_sweep.py')),
           '--train-count', str(nq), '--base-count', str(nbase), '--train-file', str(args.train_query),
           '--base-file', str(args.base), '--dev-count', str(args.dev_count),
           '--calibration-count', str(args.calibration_count), '--seed', str(args.seed),
           '--percentages', args.percentages, '--out', str(args.out / 'split')]
    subprocess.run(cmd, check=True)
    split = json.loads((args.out / 'split/manifest.json').read_text(encoding='utf-8'))
    groups = {key: np.fromfile(args.out / 'split' / split['files'][key]['file'], dtype='<u4')
              for key in ('training_order', 'development', 'calibration')}
    checks = {'status': 'not_applicable_generated_exact', 'queries': 0}
    start = time.perf_counter()
    if gt is not None and args.audit_queries:
        audit_ids = np.concatenate([value[:min(args.audit_queries, len(value))] for value in groups.values()])
        expected = exact_topk(base, queries[audit_ids], args.top_l, args.base_block, backend)
        actual = np.asarray(gt[audit_ids, :args.top_l]); validate_gt_rows(actual, nbase)
        equal = np.all(np.sort(expected, axis=1) == np.sort(actual, axis=1), axis=1)
        equal &= np.all(np.sort(expected[:, :100], axis=1) == np.sort(actual[:, :100], axis=1), axis=1)
        checks = {'status': 'passed' if equal.all() else 'failed', 'queries': len(audit_ids),
                  'failed_query_ids': audit_ids[~equal].tolist(), 'backend': backend,
                  'note': 'Sampled top-100 and top-L set audit only, not a proof for every source row.'}
        dump_json(args.out / 'source_gt_audit.json', checks)
        if not equal.all():
            raise ValueError('Training GT top-L disagrees with exhaustive search. Check mapping/metric/ties; '
                             'use --train-gt none in a NEW output directory to regenerate.')
    elif gt is not None:
        checks = {'status': 'not_audited', 'queries': 0}
    # FAISS index is built once for generation, not once per query batch.
    faiss_index = None
    if gt is None and backend == 'faiss':
        import faiss
        faiss_index = faiss.IndexFlatIP(dim)
        for offset in range(0, nbase, args.base_block):
            chunk = np.asarray(base[offset:offset + args.base_block], dtype=np.float32, order='C')
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite base data')
            faiss_index.add(chunk)
    outputs = {}
    for key, ids in groups.items():
        label = 'train' if key == 'training_order' else key
        qpath, gpath = args.out / f'{label}.fbin', args.out / f'{label}.gt.ibin'
        with qpath.open('xb') as qout, gpath.open('xb') as gout:
            qout.write(struct.pack('<II', len(ids), dim))
            gout.write(struct.pack('<II', len(ids), args.top_l))
            for offset in range(0, len(ids), args.query_batch):
                batch_ids = ids[offset:offset + args.query_batch]
                q = np.asarray(queries[batch_ids], dtype='<f4', order='C')
                if not np.isfinite(q).all():
                    raise ValueError('Nonfinite sampled query')
                if gt is not None:
                    neighbors = np.asarray(gt[batch_ids, :args.top_l], dtype='<i4', order='C')
                elif faiss_index is not None:
                    _, neighbors = faiss_index.search(q, args.top_l)
                    neighbors = neighbors.astype('<i4')
                else:
                    neighbors = exact_topk(base, q, args.top_l, args.base_block, 'numpy')
                validate_gt_rows(neighbors, nbase)
                q.tofile(qout); neighbors.tofile(gout)
                if offset == 0 or (offset // args.query_batch) % 20 == 0:
                    print(f'{label}: {min(offset + len(q), len(ids))}/{len(ids)} rows', flush=True)
        outputs[label] = {'query': str(qpath.resolve()), 'gt': str(gpath.resolve()), 'count': len(ids),
                          'query_sha256': sha256(qpath), 'gt_sha256': sha256(gpath)}
    manifest = {'completed': True, 'source': {'base': identity(args.base), 'train_query': identity(args.train_query),
                 'train_gt': identity(source_gt) if source_gt else None}, 'gt_audit': checks,
                'backend': backend, 'top_l': args.top_l, 'seed': args.seed,
                'checkpoints': split['checkpoints'], 'outputs': outputs,
                'seconds': time.perf_counter() - start,
                'test_queries_used': False, 'source_grouping_audited': False}
    dump_json(args.out / 'prepared.json', manifest)
    print(f'Prepared data: {args.out / "prepared.json"}', flush=True)


def logged(command, logfile, env=None):
    print('RUN:', ' '.join(map(str, command)), flush=True)
    with Path(logfile).open('x', encoding='utf-8') as log:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        if process.wait():
            raise RuntimeError(f'Command failed; see {logfile}')


def build_labels(args):
    info = json.loads((args.prepared / 'prepared.json').read_text(encoding='utf-8'))
    if not info['completed'] or args.out.exists():
        raise ValueError('Require completed preparation and fresh output directory')
    sizes = [p['query_count'] for p in info['checkpoints']]
    if args.max_queries:
        sizes = [x for x in sizes if x <= args.max_queries]
    if not sizes:
        raise ValueError('No checkpoints selected')
    base_audit = audit_index_base(args.index, info['source']['base']['path'])
    args.out.mkdir(parents=True)
    manifest = {'completed': False, 'index': identity(args.index), 'index_sha256': sha256(args.index),
                'prepared': str(args.prepared.resolve()), 'top_l': info['top_l'], 'checkpoints': sizes,
                'training_gt_sha256': sha256(info['outputs']['train']['gt']), 'base_index_audit': base_audit}
    if manifest['training_gt_sha256'] != info['outputs']['train']['gt_sha256']:
        raise ValueError('Prepared training GT has changed')
    dump_json(args.out / 'build.json', manifest)
    command = [args.binary, '--index', args.index, '--gt', info['outputs']['train']['gt'],
               '--out', args.out / 'marks', '--L', str(info['top_l']),
               '--checkpoints', ','.join(map(str, sizes))]
    logged(command, args.out / 'build.log')
    if identity(args.index) != manifest['index']:
        raise ValueError('Index changed during label construction')
    manifest['completed'] = True
    dump_json(args.out / 'build.json', manifest)


def search(args):
    import random
    if args.out.exists():
        raise ValueError('Choose a fresh search output directory')
    build = json.loads((args.labels / 'build.json').read_text(encoding='utf-8'))
    info = json.loads((Path(build['prepared']) / 'prepared.json').read_text(encoding='utf-8'))
    if not build['completed'] or sha256(args.index) != build['index_sha256']:
        raise ValueError('Labels do not match index or build is incomplete')
    if args.split == 'test':
        if not args.query or not args.gt:
            raise ValueError('Test requires explicit --query and --gt')
        query, gt = args.query.resolve(), args.gt.resolve()
    else:
        query, gt = (Path(info['outputs'][args.split][key]) for key in ('query', 'gt'))
        if sha256(query) != info['outputs'][args.split]['query_sha256'] or sha256(gt) != info['outputs'][args.split]['gt_sha256']:
            raise ValueError('Prepared evaluation files have changed')
    nq, _, _ = read_header(query); gn, gd, _ = read_header(gt, allow_tail=True)
    limit = args.limit or nq
    if nq != gn or gd < 100 or not 0 < limit <= nq:
        raise ValueError('Evaluation query/GT shape mismatch')
    efs = [int(v) for v in args.efs.split(',')]
    if not efs or min(efs) < 100 or len(efs) != len(set(efs)):
        raise ValueError('Distinct ef>=100 required')
    sizes = build['checkpoints'] if args.checkpoints == 'all' else [int(v) for v in args.checkpoints.split(',')]
    if not sizes or any(v not in build['checkpoints'] for v in sizes):
        raise ValueError('Unknown label checkpoint')
    prefix = [] if args.cpu == 'none' else ['taskset', '-c', args.cpu]
    if prefix and not shutil.which('taskset'):
        raise ValueError('taskset missing; use --cpu none or install affinity support')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    common = prefix + [str(args.binary.resolve()), '--index', str(args.index.resolve()),
        '--query', str(query), '--gt', str(gt), '--K', '100', '--query-limit', str(limit),
        '--query-offset', '0', '--beam-efs', args.efs, '--bound-efs', args.efs,
        '--rounds', str(args.rounds), '--verify-limit', str(min(100, limit)),
        '--timing-protocol', 'original', '--warmup-passes', '0', '--require-unrepaired', '1',
        '--seed', '17', '--cur-margin', '0', '--mid-gap', str(sys.float_info.max)]
    cases = [('unfiltered', []), ('gap', ['--mid-gap', '0.0266'])]
    for count in sizes:
        for mode in ('A', 'B'):
            cases.append((f'q{count}_{mode}', ['--marks', str((args.labels / 'marks' / f'q{count}.abm').resolve()),
                                                '--ab-mode', mode]))
    random.Random(17).shuffle(cases)
    args.out.mkdir(parents=True)
    manifest = {'completed': False, 'split': args.split, 'query': identity(query), 'gt': identity(gt),
                'index': build['index'], 'cases': [], 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'func'}}
    dump_json(args.out / 'run.json', manifest)
    for name, extra in cases:
        folder = args.out / name; folder.mkdir()
        command = common + extra + ['--out', str((folder / 'timings.csv').resolve())]
        manifest['cases'].append({'name': name, 'command': command, 'completed': False})
        dump_json(args.out / 'run.json', manifest)
        logged(command, folder / 'benchmark.log', env)
        manifest['cases'][-1]['completed'] = True
        dump_json(args.out / 'run.json', manifest)
    manifest['completed'] = True; dump_json(args.out / 'run.json', manifest)
    summarize(args.out)


def summarize(folder):
    import statistics
    import math
    records = []
    for point in sorted(Path(folder).glob('*/timings.csv')):
        with point.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            groups.setdefault((row['method'], row['ef']), []).append(row)
        diag = {}
        latencies = {}
        latency_path = Path(str(point) + '.latencies.csv')
        if latency_path.exists():
            with latency_path.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    latencies.setdefault((row['method'], row['ef']), []).append(float(row['latency_us']))
        with Path(str(point) + '.queries.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                d = diag.setdefault((row['method'], row['ef']), dict(probes=0, successful=0, rejected=0,
                    allowed=0, unknown=0, ndc_nn=0, nn_zero=0))
                for field, column in [('probes','probes'),('successful','successful_probes'),('rejected','rejected_mids'),
                                      ('allowed','mark_allowed'),('unknown','mark_unknown'),('ndc_nn','ndc_nn'),('nn_zero','nn_zero_admission')]:
                    d[field] += int(row[column])
        for (method, ef), values in groups.items():
            d = diag[(method, ef)]
            times = sorted(latencies.get((method, ef), []))
            def quantile(p):
                return times[max(0, math.ceil(p * len(times)) - 1)] if times else ''
            records.append({'case': point.parent.name, 'method': method, 'ef': ef,
                'recall': values[0]['recall'], 'mean_ndc': values[0]['mean_ndc'],
                'qps_median': statistics.median(float(v['qps']) for v in values),
                'p95_latency_us': quantile(.95), 'p99_latency_us': quantile(.99),
                'mark_allow_fraction': d['allowed']/d['rejected'] if d['rejected'] else 0,
                'unknown_fraction': d['unknown']/d['rejected'] if d['rejected'] else 0,
                'zero_admission_fraction': 1-d['successful']/d['probes'] if d['probes'] else 0,
                'zero_admission_nn_cost_share': d['nn_zero']/d['ndc_nn'] if d['ndc_nn'] else 0,
                'probes': d['probes'], 'total_nn_ndc': d['ndc_nn']})
    if not records:
        raise ValueError('No timing results found')
    with (Path(folder)/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print('Summary:', Path(folder)/'summary.csv')


def inspect(args):
    for path in args.paths:
        p = Path(path)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and any(s in child.name.lower() for s in ('train', 'gt', 'base', 'query')):
                    print(identity(child))
        elif p.is_file():
            print(identity(p))
            if p.suffix != '.index':
                try: print('header:', read_header(p, allow_tail=True))
                except ValueError as exc: print(exc)
        else: print('MISSING:', p)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s=sub.add_parser('inspect'); s.add_argument('paths', nargs='+'); s.set_defaults(func=inspect)
    s=sub.add_parser('audit-gt', help='Read-only header/ID/exhaustive-IP audit of candidate GT files')
    for name in ('base','query','out'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--gt', nargs='+', type=Path, required=True)
    s.add_argument('--samples', type=int, default=12); s.add_argument('--top-l', type=int, default=200)
    s.add_argument('--seed', type=int, default=20261019); s.add_argument('--base-block', type=int, default=32768)
    s.set_defaults(func=audit_gt)
    s=sub.add_parser('prepare')
    for name in ('base','train-query','out'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--train-gt', default='auto', help='auto, none (generate), or explicit ibin file')
    s.add_argument('--percentages', default='0.1,0.5,1')
    s.add_argument('--top-l', type=int, default=200); s.add_argument('--seed', type=int, default=17)
    s.add_argument('--dev-count', type=int, default=1000); s.add_argument('--calibration-count', type=int, default=1000)
    s.add_argument('--audit-queries', type=int, default=8)
    s.add_argument('--backend', choices=['auto','numpy','faiss'], default='auto')
    s.add_argument('--query-batch', type=int, default=32); s.add_argument('--base-block', type=int, default=32768)
    s.set_defaults(func=prepare)
    s=sub.add_parser('labels')
    for name in ('binary','index','prepared','out'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--max-queries', type=int, default=0); s.set_defaults(func=build_labels)
    s=sub.add_parser('search')
    for name in ('binary','index','labels','out'): s.add_argument('--'+name, type=Path, required=True)
    s.add_argument('--split', choices=['development','calibration','test'], default='development')
    s.add_argument('--query', type=Path); s.add_argument('--gt', type=Path)
    s.add_argument('--limit', type=int, default=0); s.add_argument('--rounds', type=int, default=3)
    s.add_argument('--cpu', default='1'); s.add_argument('--checkpoints', default='all')
    s.add_argument('--efs', default='100,200,400,800,1200,1600,2400,3200'); s.set_defaults(func=search)
    s=sub.add_parser('summarize'); s.add_argument('folder', type=Path); s.set_defaults(func=lambda a:summarize(a.folder))
    args=p.parse_args()
    for name in ('query_batch','base_block','dev_count','calibration_count','rounds','samples'):
        if hasattr(args,name) and getattr(args,name)<=0: p.error(name+' must be positive')
    if hasattr(args,'audit_queries') and args.audit_queries<0: p.error('audit_queries must be nonnegative')
    args.func(args)


if __name__ == '__main__':
    main()
