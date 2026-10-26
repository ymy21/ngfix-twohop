#!/usr/bin/env python3
"""Compare float NGFix indexes written by HNSW_NGFix::StoreIndex.

No external dependencies. Default ABI: little endian, 64-bit size_t,
32-bit node IDs, float32 vectors. Padding slots are intentionally ignored.
Exit 0: exact base snapshot match; 1: mismatch; 2: invalid/incomplete input.
This establishes structural equivalence, not historical build provenance.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


class Index:
    def __init__(self, path, size_t_bytes):
        self.path = Path(path)
        self.file = self.path.open('rb')
        self.before = self.path.stat()
        self.header_size = 7 * size_t_bytes
        values = struct.unpack('<7' + ('Q' if size_t_bytes == 8 else 'I'),
                               self.read(self.header_size))
        self.header = dict(zip(('M', 'M0', 'MEX', 'n', 'entry_point', 'dim',
                                'max_elements'), values))
        h = self.header
        if not (0 < h['n'] <= h['max_elements'] and h['dim'] > 0
                and h['entry_point'] < h['n'] and h['M0'] == 2 * h['M']):
            raise ValueError(f'{path}: invalid header or wrong ABI: {h}')
        self.stride = h['dim'] * 4 + 1
        self.vector_bytes = h['max_elements'] * self.stride
        self.graph_offset = self.header_size + self.vector_bytes
        minimum = self.graph_offset + 4 * h['n']
        if self.before.st_size < minimum:
            raise ValueError(f'{path}: incomplete or wrong format: '
                             f'{self.before.st_size} bytes; header requires at '
                             f'least {minimum} bytes (graph starts at {self.graph_offset})')
        self.stats = dict(base_edges=0, repair_edges=0, ngfix_edges=0,
                          rfix_edges=0, deleted_nodes=0)
        self.active_sha = hashlib.sha256()
        self.all_sha = hashlib.sha256()

    def read(self, count):
        data = self.file.read(count)
        if len(data) != count:
            raise ValueError(f'{self.path}: unexpected EOF at {self.file.tell()}')
        return data

    def node(self, node_id):
        cap, sz, rcap, rsz = self.read(4)
        if not (rcap <= cap and rsz <= rcap and rsz <= sz <= cap
                and sz - rsz <= cap - rcap):
            raise ValueError(f'{self.path}: invalid node metadata at {node_id}')
        raw = self.read(4 * cap)
        base = raw[4 * rcap:4 * (rcap + sz - rsz)]
        repair = raw[4 * (rcap - rsz):4 * rcap]
        for target, in struct.iter_unpack('<I', base + repair):
            if target >= self.header['n']:
                raise ValueError(f'{self.path}: invalid target {target} at {node_id}')
        ehs = self.read(2 * rcap)
        active_ehs = ehs[2 * (rcap - rsz):]
        rfix = sum(eh == 201 for eh, in struct.iter_unpack('<H', active_ehs))
        self.stats['base_edges'] += sz - rsz
        self.stats['repair_edges'] += rsz
        self.stats['rfix_edges'] += rfix
        self.stats['ngfix_edges'] += rsz - rfix
        return base

    def finish(self):
        if self.file.read(1):
            raise ValueError(f'{self.path}: trailing bytes after declared nodes')
        self.stable()

    def stable(self):
        after = self.path.stat()
        if (after.st_size, after.st_mtime_ns) != (self.before.st_size, self.before.st_mtime_ns):
            raise ValueError(f'{self.path}: file changed during verification')


def compare(args):
    a, b = Index(args.base, args.size_t_bytes), Index(args.repaired, args.size_t_bytes)
    report = {'base_path': str(a.path), 'repaired_path': str(b.path),
              'base_header': a.header, 'repaired_header': b.header,
              'format': 'little-endian float32; size_t=' + str(args.size_t_bytes)}
    # Allocation size may differ; active IDs and traversal parameters must match.
    fields = ('M', 'M0', 'MEX', 'n', 'entry_point', 'dim')
    report['matching_header'] = all(a.header[k] == b.header[k] for k in fields)
    if not report['matching_header']:
        a.stable()
        b.stable()
        report.update(verified=False, reason='header mismatch; no full graph comparison')
        return report
    # Hash active records (including deletion flags) in node-ID order.
    for idx in (a, b):
        remaining = idx.vector_bytes
        active_remaining = idx.header['n'] * idx.stride
        block_size = max(1, (8 * 1024 * 1024) // idx.stride) * idx.stride
        while remaining:
            block = idx.read(min(remaining, block_size))
            idx.all_sha.update(block)
            active = block[:min(active_remaining, len(block))]
            idx.active_sha.update(active)
            idx.stats['deleted_nodes'] += sum(active[::idx.stride])
            active_remaining -= len(active)
            remaining -= len(block)
    order_mismatch = set_mismatch = 0
    examples = []
    for node_id in range(a.header['n']):
        left, right = a.node(node_id), b.node(node_id)
        if left != right:
            order_mismatch += 1
            # Sorted lists also preserve duplicate multiplicity.
            li = [v for v, in struct.iter_unpack('<I', left)]
            ri = [v for v, in struct.iter_unpack('<I', right)]
            if sorted(li) != sorted(ri):
                set_mismatch += 1
            if len(examples) < 5:
                examples.append({'node_id': node_id, 'base': li, 'repaired_base': ri})
    a.finish()
    b.finish()
    vector_match = a.active_sha.hexdigest() == b.active_sha.hexdigest()
    pure = a.stats['repair_edges'] == 0
    report.update(
        active_vector_records_match=vector_match,
        base_active_vector_sha256=a.active_sha.hexdigest(),
        repaired_active_vector_sha256=b.active_sha.hexdigest(),
        base_all_vector_sha256=a.all_sha.hexdigest(),
        repaired_all_vector_sha256=b.all_sha.hexdigest(),
        base_stats=a.stats, repaired_stats=b.stats,
        base_is_unrepaired=pure,
        base_neighbor_order_mismatch_nodes=order_mismatch,
        base_neighbor_multiset_mismatch_nodes=set_mismatch,
        mismatch_examples=examples,
        verified=vector_match and pure and order_mismatch == 0,
        provenance_note='Structural equivalence does not prove the historical build command.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--repaired', required=True)
    parser.add_argument('--size-t-bytes', type=int, choices=(4, 8), default=8)
    parser.add_argument('--output', help='JSON report path')
    args = parser.parse_args()
    try:
        report = compare(args)
        code = 0 if report['verified'] else 1
    except (OSError, ValueError, struct.error) as exc:
        report = {'verified': False, 'error': str(exc)}
        code = 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + '\n', encoding='utf-8')
    return code


if __name__ == '__main__':
    sys.exit(main())
