import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from analyze_hnsw_stage1 import summarize, match_targets, paired_groups
from compare_graph_k_matrix import check_pair, target_table

ROOT=Path(__file__).resolve().parents[1]


class Stage1Tests(unittest.TestCase):
    def fixture(self):
        m={'parameters':dict(efs=[100,200],rounds=2,k=100,offset=0,limit=2,verify_limit=2,min_batch_ms=10)}
        timings=[];queries=[]
        for method,ef,recall,qps,ndc in [('beam',100,.90,1000,100),('beam',200,.95,600,200),
                                        ('bound_unbounded',100,.95,800,150),('bound_unbounded',200,.99,400,300)]:
            for rnd in range(2):
                timings.append(dict(method=method,ef=ef,round=rnd,k=100,query_offset=0,queries=2,
                                    verified_queries=2,qps=qps,seconds=.02,batch_queries=int(qps*.02),
                                    recall=recall,mean_ndc=ndc))
            for qid in range(2):
                queries.append(dict(method=method,ef=ef,query_id=qid,recall=recall,ndc=ndc))
        return timings,queries,m

    def test_recall_target_matches_different_efs(self):
        t,q,m=self.fixture();points=summarize(t,q,m)
        row=match_targets(points,[.95])[0]
        self.assertEqual(row['beam_ef'],200)
        self.assertEqual(row['bound_unbounded_ef'],100)
        self.assertAlmostEqual(row['bound_over_beam_qps'],800/600)
        self.assertAlmostEqual(row['bound_over_beam_ndc'],.75)
        self.assertFalse(row['needs_denser_ef_grid'])

    def test_under_target_is_unreached(self):
        t,q,m=self.fixture();row=match_targets(summarize(t,q,m),[.995])[0]
        self.assertFalse(row['beam_reached']);self.assertFalse(row['bound_unbounded_reached'])
        self.assertEqual(row['bound_over_beam_qps'],'')

    def test_original_protocol_uses_query_latency_sum_not_min_batch(self):
        t,q,m=self.fixture();m['parameters']['timing_protocol']='original'
        for row in t:
            row['timing_protocol']='original';row['batch_queries']=2;row['seconds']=2/row['qps']
        self.assertEqual(len(summarize(t,q,m)),4)
        t[0]['timing_protocol']='batch'
        with self.assertRaises(ValueError):summarize(t,q,m)

    def test_original_plan_disables_explicit_warmup(self):
        folder=ROOT/'output/stage1_tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'analysis/run_hnsw_stage1.py'),
                            '--bin','new_binary','--index','base','--query','q','--gt','gt',
                            '--out',str(out),'--timing-protocol','original','--dry-run'],check=True,capture_output=True)
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(m['parameters']['warmup_passes'],0)
            self.assertEqual(m['command'][m['command'].index('--timing-protocol')+1],'original')

    def test_missing_duplicate_inconsistent_data_rejected(self):
        t,q,m=self.fixture()
        with self.assertRaises(ValueError):summarize(t[:-1],q,m)
        with self.assertRaises(ValueError):summarize(t,q+[q[0]],m)
        bad=copy.deepcopy(q);bad[0]['ndc']=999
        with self.assertRaises(ValueError):summarize(t,bad,m)

    def test_fixed_group_is_reused_across_efs(self):
        t,q,m=self.fixture();rows=paired_groups(q,200)
        hard=[r for r in rows if r['group']=='fixed_low_recall_20pct']
        self.assertEqual([r['queries'] for r in hard],[1,1])
        self.assertTrue(all(r['anchor_ef']==200 for r in hard))

    def test_plan_uses_independent_binary_no_cost_limits(self):
        folder=ROOT/'output/stage1_tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp)/'plan'
            subprocess.run([sys.executable,str(ROOT/'analysis/run_hnsw_stage1.py'),
                            '--bin','new_binary','--index','base','--query','q','--gt','gt',
                            '--out',str(out),'--dry-run'],check=True,capture_output=True)
            m=json.loads((out/'manifest.json').read_text())
            self.assertFalse(m['completed'])
            self.assertEqual(m['profile']['budgets'],'unlimited')
            self.assertEqual(m['profile']['nn_selection'],'full')
            self.assertIn('--bound-efs',m['command'])
            self.assertNotIn('boundtwohopcmp',m['command'])
            self.assertNotIn('--event_trace_ef',m['command'])

    def test_matrix_config_mismatch_rejected(self):
        import struct
        t,q,m=self.fixture()
        a=dict(suite='hnsw_stage1',completed=True,graph='unrepaired_bottom',profile={},
               binary_sha256='same',platform='same',parameters=m['parameters'].copy(),
               inputs=dict(query={'sha256':'q'},gt={'sha256':'gt'},
                           index={'header_hex':struct.pack('<7Q',16,32,48,256,0,64,256).hex()}))
        a['parameters'].update(warmup_passes=1,seed=1,cpu='1')
        b=copy.deepcopy(a);b['graph']='repaired_reference'
        check_pair(a,b)
        for field in ('k','cpu','efs'):
            bad=copy.deepcopy(b);bad['parameters'][field]='different'
            with self.assertRaises(ValueError):check_pair(a,bad)
        bad=copy.deepcopy(b);bad['inputs']['query']['sha256']='different'
        with self.assertRaises(ValueError):check_pair(a,bad)

    def test_matrix_target_comparison_uses_four_curves(self):
        t,q,m=self.fixture();points=summarize(t,q,m)
        curves=[dict(graph=graph,**point) for graph in ('hnsw','ngfix') for point in points]
        row=target_table(curves,[.95])[0]
        self.assertEqual(row['hnsw_beam_ef'],200)
        self.assertEqual(row['ngfix_bound_unbounded_ef'],100)
        self.assertAlmostEqual(row['hnsw_bound_qps_over_ngfix_beam'],800/600)

    def test_completed_matrix_cli(self):
        import struct
        folder=ROOT/'output/stage1_tests';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            root=Path(tmp)
            for k in (10,100):
                for graph,label in [('hnsw','unrepaired_bottom'),('ngfix','repaired_reference')]:
                    timings,queries,fixture=self.fixture()
                    for r in timings:r['k']=k
                    params=fixture['parameters'];params.update(k=k,warmup_passes=1,seed=1,cpu='1')
                    m=dict(suite='hnsw_stage1',completed=True,graph=label,profile={},binary_sha256='same',platform='same',
                           parameters=params,inputs=dict(query={'sha256':'q'},gt={'sha256':'gt'},index={
                               'path':str(root/('base.index' if graph=='hnsw' else 'repaired.index')),
                               'header_hex':struct.pack('<7Q',16,32,48,256,0,64,256).hex()}))
                    run=root/(graph+'_k'+str(k));clean=run/'clean';clean.mkdir(parents=True)
                    (clean/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
                    for name,items in [('timings.csv',timings),('timings.csv.queries.csv',queries)]:
                        with (clean/name).open('w',newline='',encoding='utf-8') as f:
                            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
                    (run/'index_pair_verification.json').write_text(json.dumps(dict(verified=True,
                        base_path=str(root/'base.index'),repaired_path=str(root/'repaired.index'))),encoding='utf-8')
            result=subprocess.run([sys.executable,str(ROOT/'analysis/compare_graph_k_matrix.py'),
                                   '--indir',str(root),'--targets','.95'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with (root/'comparison/k10_at_target_recall.csv').open(newline='',encoding='utf-8') as f:
                row=next(csv.DictReader(f))
            self.assertEqual(row['hnsw_beam_ef'],'200')
            self.assertEqual(row['ngfix_bound_unbounded_ef'],'100')


if __name__=='__main__':unittest.main()
