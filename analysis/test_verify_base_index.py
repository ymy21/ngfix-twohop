"""Small binary fixtures exercise the real storage boundaries and failure gates."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('verify_base_index.py')


def write_index(path, repaired=False, reverse=False, changed_vector=False):
    data = struct.pack('<7Q', 16, 32, 48, 3, 0, 2, 3)
    for i in range(3):
        data += b'\0' + struct.pack('<2f', i, 9 if changed_vector and i == 1 else i + 1)
    for i in range(3):
        base = [(i + 1) % 3, (i + 2) % 3]
        if reverse and i == 0:
            base.reverse()
        if repaired:
            # Two empty repair slots, then NGFix and RFix, then base slots.
            data += bytes((8, 4, 4, 2))
            data += struct.pack('<8I', 0xffffffff, 0xffffffff, (i + 1) % 3,
                                (i + 2) % 3, *base, 0xffffffff, 0xffffffff)
            data += struct.pack('<4H', 0xffff, 0xffff, 200, 201)
        else:
            data += bytes((4, 2, 0, 0))
            data += struct.pack('<4I', *base, 0xffffffff, 0xffffffff)
    path.write_bytes(data)


class VerificationTests(unittest.TestCase):
    def run_pair(self, *, base_repaired=False, reverse=False, changed_vector=False,
                 truncate=False, trailing=False):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'base.index', Path(temp) / 'repaired.index'
            write_index(a, repaired=base_repaired)
            write_index(b, repaired=True, reverse=reverse, changed_vector=changed_vector)
            if truncate:
                b.write_bytes(b.read_bytes()[:-1])
            if trailing:
                b.write_bytes(b.read_bytes() + b'bad')
            result = subprocess.run([sys.executable, str(SCRIPT), '--base', str(a),
                                     '--repaired', str(b)], capture_output=True, text=True)
            return result.returncode, json.loads(result.stdout)

    def test_matching_active_data_and_base_edges_ignore_padding(self):
        code, report = self.run_pair()
        self.assertEqual(code, 0)
        self.assertEqual(report['repaired_stats']['repair_edges'], 6)
        self.assertEqual(report['repaired_stats']['rfix_edges'], 3)
        self.assertEqual(report['base_neighbor_order_mismatch_nodes'], 0)

    def test_same_edge_set_different_order_is_reported_and_rejected(self):
        code, report = self.run_pair(reverse=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['base_neighbor_order_mismatch_nodes'], 1)
        self.assertEqual(report['base_neighbor_multiset_mismatch_nodes'], 0)

    def test_vector_change_is_rejected(self):
        code, report = self.run_pair(changed_vector=True)
        self.assertEqual(code, 1)
        self.assertFalse(report['active_vector_records_match'])

    def test_repaired_base_is_rejected(self):
        code, report = self.run_pair(base_repaired=True)
        self.assertEqual(code, 1)
        self.assertFalse(report['base_is_unrepaired'])

    def test_truncation_and_trailing_bytes_are_rejected(self):
        for kwargs in ({'truncate': True}, {'trailing': True}):
            with self.subTest(kwargs=kwargs):
                code, report = self.run_pair(**kwargs)
                self.assertEqual(code, 2)
                self.assertIn('error', report)


if __name__ == '__main__':
    unittest.main()
