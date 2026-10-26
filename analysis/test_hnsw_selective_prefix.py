import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from analyze_hnsw_selective_prefix import summarize_case,attach_latencies
from run_hnsw_selective_prefix import profiles

ROOT=Path(__file__).resolve().parents[1]
TEMP_ROOT=ROOT/'output/selective_prefix_tests'
TEMP_ROOT.mkdir(parents=True,exist_ok=True)


class SelectivePrefixTests(unittest.TestCase):
    def fixture(self):
        p=dict(k=10,limit=2,offset=1000,rounds=2,verify_limit=2,timing_protocol='original',
            min_batch_ms=300,nn_cap=8,nn_limit=4,mid_cap=2)
        case=dict(name='cur_mid',beam_efs=[20],other_efs=[20],other_method='selective_prefix')
        timings=[];queries=[]
        for method,recall,ndc in [('beam',.9,50),('selective_prefix',1.0,100)]:
            for rnd in range(2):
                timings.append(dict(method=method,ef=20,round=rnd,k=10,queries=2,query_offset=1000,
                    verified_queries=2,qps=1000,seconds=.002,batch_queries=2,timing_protocol='original',recall=recall,mean_ndc=ndc))
            for qid in (1000,1001):
                r=dict(method=method,ef=20,query_id=qid,recall=recall,ndc=ndc)
                if method=='selective_prefix':r.update(ndc_entry=1,ndc_mid=91,ndc_nn=8,rejected_mids=10,
                    probes=3,blocked_cur=3,blocked_gap=2,blocked_budget=1,blocked_selection=1,
                    successful_probes=2,admitted_nn=3,scanned_nn=12,expanded=2)
                queries.append(r)
        return case,timings,queries,p

    def test_valid_separate_counters_and_original_timing(self):
        case,t,q,p=self.fixture();rows=summarize_case(case,t,q,p)
        self.assertEqual(len(rows),2);self.assertEqual(rows[1]['mean_ndc'],100)

    def test_missing_duplicate_and_counter_budget_errors(self):
        c,t,q,p=self.fixture()
        with self.assertRaises(ValueError):summarize_case(c,t[:-1],q,p)
        with self.assertRaises(ValueError):summarize_case(c,t,q+[q[0]],p)
        bad=copy.deepcopy(q);bad[-1]['blocked_gap']=1
        with self.assertRaises(ValueError):summarize_case(c,t,bad,p)
        bad=copy.deepcopy(p);bad['nn_cap']=7
        with self.assertRaises(ValueError):summarize_case(c,t,q,bad)
        bad=copy.deepcopy(t);bad[0]['qps']=999
        with self.assertRaises(ValueError):summarize_case(c,bad,q,p)

    def test_ablations_change_only_distance_gates(self):
        rows=profiles(.0266,.0127)
        self.assertEqual([x[0] for x in rows],['caps_only','mid_gap','cur_mid'])
        self.assertEqual(rows[1][1],rows[2][1]);self.assertEqual(rows[1][2],0)

    def test_latency_population_and_summary(self):
        c,t,q,p=self.fixture();points=summarize_case(c,t,q,p)
        rows=[dict(method=r['method'],ef=r['ef'],round=rnd,query_id=r['query_id'],latency_us=1000,ndc=r['ndc'])
            for r in q for rnd in range(2)]
        attach_latencies(points,rows,q,p)
        self.assertEqual(points[1]['latency_p95_us'],1000)
        with self.assertRaises(ValueError):attach_latencies(points,rows[:-1],q,p)

    def test_full_summary_writes_expected_comparisons(self):
        case,t,q,p=self.fixture()
        def write(path,rows):
            with path.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader();w.writerows(rows)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            folder=Path(tmp);cases=[]
            for name in ('caps_only','mid_gap','cur_mid'):
                c=dict(case,name=name);cases.append(c);d=folder/name;d.mkdir()
                write(d/'timings.csv',t);write(d/'timings.csv.queries.csv',q)
                write(d/'timings.csv.latencies.csv',[dict(method=r['method'],ef=r['ef'],round=rnd,
                    query_id=r['query_id'],latency_us=1000,ndc=r['ndc']) for r in q for rnd in range(2)])
            (folder/'manifest.json').write_text(json.dumps(dict(completed=True,cases=cases,parameters=p)),encoding='utf-8')
            subprocess.run([sys.executable,str(ROOT/'analysis/analyze_hnsw_selective_prefix.py'),'--indir',str(folder)],check=True,capture_output=True)
            with (folder/'analysis/target_recall.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            r=next(r for r in rows if r['case']=='cur_mid' and float(r['target_recall'])==.95)
            self.assertEqual(r['beam_reached'],'False');self.assertEqual(r['method_reached'],'True')
            self.assertEqual(r['method_over_beam_qps'],'')

    def test_plan_retains_old_control_and_no_gt_in_search_options(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'analysis/run_hnsw_selective_prefix.py'),
                '--bin','new','--unbounded-bin','old','--query','q','--gt','gt','--index','base',
                '--out',str(out),'--dry-run'],check=True,capture_output=True)
            m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            self.assertFalse(m['completed']);self.assertEqual(m['parameters']['offset'],1000)
            self.assertEqual(len(m['cases']),4)
            for case in m['cases']:
                cmd=case['command']
                self.assertEqual(cmd[cmd.index('--warmup-passes')+1],'0')
                self.assertEqual(cmd[cmd.index('--require-unrepaired')+1],'1')
                self.assertNotIn('--event_trace_ef',cmd)
            cmd=m['cases'][-1]['command']
            self.assertEqual(cmd[cmd.index('--mid-gap')+1],'0.0266')
            self.assertEqual(cmd[cmd.index('--cur-margin')+1],'0.0127')

    def test_invalid_subgrid_does_not_create_output(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as tmp:
            out=Path(tmp)/'bad'
            r=subprocess.run([sys.executable,str(ROOT/'analysis/run_hnsw_selective_prefix.py'),
                '--bin','new','--unbounded-bin','old','--query','q','--gt','gt','--index','base',
                '--out',str(out),'--efs','100,200','--unbounded-efs','400','--dry-run'],capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
