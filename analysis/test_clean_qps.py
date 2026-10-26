import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from analyze_clean_qps import summarize,at_targets,match_cheap_points
ROOT=Path(__file__).resolve().parents[1]

class CleanQPSTests(unittest.TestCase):
    def fixture(self):
        m=dict(parameters=dict(beam_efs=[100,200],cheap_efs=[150],k=100,offset=1000,limit=2,rounds=2,min_batch_ms=10,verify_limit=200))
        rows=[]
        for method,ef,recall,qps in [('beam',100,.96,[1000,1100]),('beam',200,.9699999999999998,[500,550]),
                                    ('cheap32_nn8',150,.97,[600,620])]:
            for round_,speed in enumerate(qps):
                rows.append(dict(method=method,ef=ef,k=100,query_offset=1000,queries=2,round=round_,
                    batch_queries=20,seconds=20/speed,qps=speed,recall=recall,verified_queries=2,checksum='1'))
        return rows,m

    def test_threshold_and_median(self):
        rows,m=self.fixture(); points=summarize(rows,m)
        result=at_targets(points,[.97,.98])
        self.assertEqual(result[0]['beam_ef'],200)
        self.assertEqual(result[0]['cheap_ef'],150)
        self.assertAlmostEqual(result[0]['qps_ratio_cheap_over_beam'],610/525)
        self.assertFalse(result[1]['beam_reached']);self.assertFalse(result[1]['cheap_reached'])
        self.assertEqual(result[1]['qps_ratio_cheap_over_beam'],'')
        matched=match_cheap_points(points)[0]
        self.assertEqual(matched['beam_ef'],200)

    def test_under_target_is_not_accepted(self):
        rows,m=self.fixture()
        for row in rows:
            if row['method']=='cheap32_nn8': row['recall']=.9695
        points=summarize(rows,m)
        self.assertFalse(at_targets(points,[.97])[0]['cheap_reached'])
        self.assertEqual(match_cheap_points(points)[0]['beam_ef'],200)

    def test_incomplete_or_invalid_measurements_rejected(self):
        rows,m=self.fixture()
        for mutate in ['missing_round','wrong_population','bad_qps','unverified']:
            bad=copy.deepcopy(rows)
            if mutate=='missing_round': bad.pop()
            elif mutate=='wrong_population': bad[0]['query_offset']=1001
            elif mutate=='bad_qps': bad[0]['qps']=5
            else: bad[0]['verified_queries']=1
            with self.assertRaises(ValueError): summarize(bad,m)

    def test_one_process_plan(self):
        folder=ROOT/'output/clean_qps_20261008/tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'analysis/run_clean_qps.py'),'--bin','binary',
                '--query','q','--gt','gt','--index','index','--out',str(out),'--dry-run'],check=True,capture_output=True)
            m=json.loads((out/'manifest.json').read_text())
            self.assertFalse(m['completed'])
            self.assertEqual(m['profile']['cheap_dims'],32)
            self.assertEqual(m['profile']['nn_limit'],8)
            command=m['command']
            self.assertIn('--beam-efs',command);self.assertIn('--cheap-efs',command)
            self.assertNotIn('boundtwohopcmp',command);self.assertNotIn('--event_trace_ef',command)
            self.assertNotIn('--nn_trace',command)
            self.assertEqual(m['parameters']['offset'],0)
            self.assertEqual(m['parameters']['limit'],10000)
            self.assertEqual(m['parameters']['verify_limit'],200)

    def test_sampled_verification_keeps_full_population(self):
        rows,m=self.fixture()
        m['parameters']['limit']=10000
        for row in rows:
            row['queries']=10000;row['verified_queries']=200
            row['batch_queries']=10000;row['seconds']=10000/row['qps']
        points=summarize(rows,m)
        self.assertTrue(all(p['queries']==10000 for p in points))

    def test_completed_analysis_cli(self):
        folder=ROOT/'output/clean_qps_20261008/tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp);rows,m=self.fixture()
            m.update(suite='clean_qps',completed=True)
            (out/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
            with (out/'timings.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            command=[sys.executable,str(ROOT/'analysis/analyze_clean_qps.py'),'--indir',str(out),'--targets','.97,.98']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with (out/'analysis/matched_to_cheap_recall.csv').open(newline='',encoding='utf-8') as f:
                matched=list(csv.DictReader(f))
            self.assertEqual(matched[0]['beam_ef'],'200')
            self.assertAlmostEqual(float(matched[0]['qps_ratio_cheap_over_beam']),610/525)
            m['completed']=False;(out/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)

if __name__=='__main__': unittest.main()
