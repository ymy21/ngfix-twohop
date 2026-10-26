import argparse
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import ab_stage1 as ab

TEST_ROOT = Path(__file__).resolve().parent.parent / 'result' / 'ab_test_tmp'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


def write_matrix(path, values, dtype):
    a = np.asarray(values, dtype=dtype)
    with Path(path).open('wb') as stream:
        stream.write(struct.pack('<II', *a.shape)); a.tofile(stream)


class ABPreparationTests(unittest.TestCase):
    def test_block_exact_matches_full_and_ties(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(29, 7)).astype('f4')
        base[7] = base[3]
        queries = rng.normal(size=(5, 7)).astype('f4'); queries[0] = 0
        scores = queries.astype('f8') @ base.astype('f8').T
        expected = np.asarray([np.lexsort((np.arange(len(base)), -row))[:8] for row in scores])
        for block in (1, 3, 11, 100):
            np.testing.assert_array_equal(ab.exact_topk(base, queries, 8, block), expected)

    def test_gt_validation(self):
        ab.validate_gt_rows(np.array([[0, 2], [2, 1]]), 3)
        for rows in ([[0, 0]], [[-1, 0]], [[0, 3]]):
            with self.assertRaises(ValueError): ab.validate_gt_rows(np.array(rows), 3)

    def test_headers_and_discovery(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            root=Path(temp); base=root/'base.fbin'; query=root/'query.train.fbin'
            write_matrix(base, np.zeros((10, 2)), '<f4')
            write_matrix(query, np.zeros((5, 2)), '<f4')
            self.assertIsNone(ab.discover_gt(base, query, 5, 3))
            gt=root/'gt.train.top500.bin'; write_matrix(gt, np.tile(np.arange(3), (5, 1)), '<i4')
            self.assertEqual(ab.discover_gt(base, query, 5, 3), gt.resolve())
            with gt.open('ab') as f: f.write(b'extra')
            self.assertEqual(ab.read_header(gt, allow_tail=True), (5, 3, 5))
            with self.assertRaises(ValueError): ab.read_header(gt)
            gt.write_bytes(gt.read_bytes()[:10])
            with self.assertRaises(ValueError): ab.read_header(gt, allow_tail=True)

    def prepare_case(self, existing_gt):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            root=Path(temp); rng=np.random.default_rng(91)
            base=rng.normal(size=(128,512)).astype('f4')
            query=rng.normal(size=(64,512)).astype('f4')
            base_path=root/'base.fbin'; query_path=root/'query.train.fbin'
            write_matrix(base_path,base,'<f4'); write_matrix(query_path,query,'<f4')
            expected=ab.exact_topk(base,query,100,32)
            if existing_gt: write_matrix(root/'gt.train.top500.bin',expected,'<i4')
            out=root/'prepared'
            args=argparse.Namespace(base=base_path,train_query=query_path,out=out,
                train_gt='auto',top_l=100,backend='numpy',percentages='5,10',seed=17,
                dev_count=2,calibration_count=2,audit_queries=1,base_block=32,query_batch=4)
            ab.prepare(args)
            info=json.loads((out/'prepared.json').read_text(encoding='utf-8'))
            split=json.loads((out/'split/manifest.json').read_text(encoding='utf-8'))
            self.assertTrue(info['completed']); self.assertFalse(info['test_queries_used'])
            self.assertEqual([p['query_count'] for p in info['checkpoints']], [6,13])
            seen=set()
            for group,label in [('training_order','train'),('development','development'),('calibration','calibration')]:
                ids=np.fromfile(out/'split'/split['files'][group]['file'],dtype='<u4')
                self.assertFalse(seen.intersection(ids.tolist())); seen.update(ids.tolist())
                result=np.memmap(info['outputs'][label]['gt'],dtype='<i4',mode='r',offset=8,shape=(len(ids),100))
                np.testing.assert_array_equal(result,expected[ids]); del result
            self.assertEqual(info['gt_audit']['status'], 'passed' if existing_gt else 'not_applicable_generated_exact')
            with self.assertRaises(ValueError): ab.prepare(args)  # no accidental overwrite

    def test_prepare_reuses_audited_gt(self): self.prepare_case(True)
    def test_prepare_generates_missing_gt(self): self.prepare_case(False)

    def test_gt_audit_detects_permuted_rows_and_short_width(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            root = Path(temp); rng = np.random.default_rng(12)
            base = rng.normal(size=(23, 8)).astype('f4')
            query = rng.normal(size=(5, 8)).astype('f4')
            write_matrix(root/'base.fbin', base, '<f4'); write_matrix(root/'query.fbin', query, '<f4')
            exact = ab.exact_topk(base, query, 7)
            write_matrix(root/'good.ibin', exact, '<i4')
            write_matrix(root/'bad.ibin', np.roll(exact, 1, axis=0), '<i4')
            write_matrix(root/'narrow.ibin', exact[:, :3], '<i4')
            args = argparse.Namespace(base=root/'base.fbin', query=root/'query.fbin', out=root/'audit.json',
                gt=[root/'good.ibin', root/'bad.ibin', root/'narrow.ibin'], top_l=7, seed=17, samples=5, base_block=8)
            ab.audit_gt(args)
            results = json.loads(args.out.read_text(encoding='utf-8'))['candidates']
            self.assertEqual([r['status'] for r in results], ['passed', 'mismatch', 'matched_but_too_narrow'])
            args.gt = [root/'bad.ibin']; args.out = root/'failed.json'
            with self.assertRaises(ValueError): ab.audit_gt(args)
            self.assertTrue(args.out.exists())

    def test_raw_index_base_audit(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temp:
            root = Path(temp); base = np.arange(60, dtype='f4').reshape(10, 6)
            write_matrix(root/'base.fbin', base, '<f4')
            index = root/'graph.index'
            with index.open('wb') as stream:
                stream.write(struct.pack('<7Q', 16, 32, 48, 10, 0, 6, 10))
                for vector in base:
                    stream.write(b'\x00'); vector.tofile(stream)
                stream.write(bytes(4 * len(base)))
            self.assertEqual(ab.audit_index_base(index, root/'base.fbin')['status'], 'passed')
            base[0, 0] = -1; write_matrix(root/'base.fbin', base, '<f4')
            with self.assertRaisesRegex(ValueError, 'ID-order mismatch'): ab.audit_index_base(index, root/'base.fbin')


if __name__ == '__main__': unittest.main()
