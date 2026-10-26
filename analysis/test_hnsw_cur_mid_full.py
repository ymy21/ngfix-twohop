import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from analyze_hnsw_cur_mid_full import summarize_case,write
from run_hnsw_cur_mid_full import profiles,valid_graph_config

ROOT=Path(__file__).resolve().parents[1]
TEMP=ROOT/'output/cur_mid_tests'
TEMP.mkdir(parents=True,exist_ok=True)


class CurMidTests(unittest.TestCase):
    def test_primary_graph_can_be_m32(self):
        self.assertTrue(valid_graph_config('index',16,32,48))
        self.assertTrue(valid_graph_config('index',32,64,48))
        self.assertTrue(valid_graph_config('m32_index',32,64,48))
        self.assertFalse(valid_graph_config('ngfix_index',32,64,48))
        self.assertFalse(valid_graph_config('index',32,32,48))

    def fixture(self):
        params=dict(k=10,limit=1,offset=1000,rounds=2,verify_limit=1,
                    timing_protocol='original',min_batch_ms=300,mid_cap=0)
        case=dict(name='cur_mid',beam_efs=[20],other_efs=[20],other_method='cur_mid_full')
        timings=[];queries=[];latencies=[]
        for method,recall,ndc in [('beam',.9,8000),('cur_mid_full',1.,10000)]:
            for rnd in range(2):
                timings.append(dict(method=method,ef=20,round=rnd,k=10,queries=1,query_offset=1000,
                    verified_queries=1,qps=1000,seconds=.001,batch_queries=1,timing_protocol='original',
                    recall=recall,mean_ndc=ndc))
                latencies.append(dict(method=method,ef=20,round=rnd,query_id=1000,latency_us=1000,ndc=ndc))
            q=dict(method=method,ef=20,query_id=1000,recall=recall,ndc=ndc)
            if method=='cur_mid_full':q.update(ndc_entry=1,ndc_mid=7999,ndc_nn=2000,rejected_mids=10,
                probes=3,blocked_cur=3,blocked_gap=2,blocked_ratio=1,blocked_selection=1,
                successful_probes=2,admitted_nn=3,scanned_nn=2400,expanded=2)
            queries.append(q)
        for field in queries[1]:queries[0].setdefault(field,0)
        return params,case,timings,queries,latencies

    def test_no_nn_cap(self):
        p,c,t,q,_=self.fixture()
        self.assertEqual(len(summarize_case(c,t,q,p)),2) # NN=2000 exceeds old caps.

    def test_partition_and_distances(self):
        p,c,t,q,_=self.fixture()
        for field in ('ndc_nn','blocked_ratio','rejected_mids'):
            broken=copy.deepcopy(q);broken[1][field]+=1
            with self.assertRaises(ValueError):summarize_case(c,t,broken,p)

    def test_optional_mid_cap(self):
        p,c,t,q,_=self.fixture();p['mid_cap']=1
        with self.assertRaises(ValueError):summarize_case(c,t,q,p)

    def test_profiles_and_dry_run(self):
        rules=profiles(.0266,.0127)
        self.assertEqual(rules['cur_mid'],(.0266,.0127,sys.float_info.max))
        self.assertEqual(rules['ratio_1'],(sys.float_info.max,.0127,1.))
        with tempfile.TemporaryDirectory(dir=TEMP) as td:
            out=Path(td)/'plan'
            subprocess.run([sys.executable,str(ROOT/'analysis/run_hnsw_cur_mid_full.py'),
                '--bin','fake','--index','m16.index','--m32-index','m32.index','--ngfix-index','ngfix.index',
                '--query','query.fbin','--gt','gt.ibin','--out',str(out),'--k','10','--efs','10,20','--dry-run'],check=True,capture_output=True)
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(len(m['cases']),5)
            self.assertEqual(m['parameters']['mid_cap'],0)
            for c in m['cases']:
                cmd=c['command'];self.assertNotIn('--nn-cap',cmd);self.assertNotIn('--nn-limit',cmd)
                if c['name'] in ('hnsw_m32','ngfix_m16'):
                    self.assertEqual(c['other_efs'],[]);self.assertIn('--beam-only',cmd)

    def test_summary_with_cross_graph_baselines(self):
        p,c,t,q,l=self.fixture()
        with tempfile.TemporaryDirectory(dir=TEMP) as td:
            out=Path(td);cases=[c]
            for name in ('hnsw_m32','ngfix_m16'):
                cases.append(dict(name=name,beam_efs=[20],other_efs=[],other_method='cur_mid_full'))
            for case in cases:
                folder=out/case['name'];folder.mkdir()
                keep=lambda rows:[r for r in rows if case['other_efs'] or r['method']=='beam']
                write(folder/'timings.csv',keep(t));write(folder/'timings.csv.queries.csv',keep(q))
                write(folder/'timings.csv.latencies.csv',keep(l))
            (out/'manifest.json').write_text(json.dumps(dict(completed=True,parameters=p,cases=cases)))
            subprocess.run([sys.executable,str(ROOT/'analysis/analyze_hnsw_cur_mid_full.py'),'--indir',str(out)],check=True,capture_output=True)
            with (out/'analysis/target_recall.csv').open(newline='') as f:targets=list(csv.DictReader(f))
            m32=[r for r in targets if r['case']=='hnsw_m32' and float(r['target_recall'])==.9][0]
            self.assertEqual(m32['compared_method'],'beam');self.assertEqual(m32['method_reached'],'True')


if __name__=='__main__':unittest.main()
