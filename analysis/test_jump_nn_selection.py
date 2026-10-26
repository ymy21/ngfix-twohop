"""Synthetic checks for label/censoring logic and reviewable four-way run plans."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd
from analyze_jump_nn_selection import diagnose,retention_summary
ROOT=Path(__file__).resolve().parents[1]

class NNTests(unittest.TestCase):
    def test_analyzer_end_to_end(self):
        folder=ROOT/'output/nn_selection_20261008/tests'; folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp)
            names=['baseline','nn_full','nn_prefix8','nn_random8_s17','nn_cheap8_d16']
            manifest=dict(suite='nn_online',inputs={},dimension=512,
                controls_settings=dict(offset=1000,limit=2,k=100,efs=[100]),
                parameters=dict(offset=1000,limit=2,nn_limit=8,nn_cap=4000),
                cases=[dict(name=n) for n in names]+[dict(name='dabs')])
            for name in names:
                rows=[]
                for qid in (1000,1001):
                    nn=0 if name=='baseline' else 10 if name=='nn_full' else 8
                    row=dict(row_type='QUERY',ef=100,query_id=qid,ndc=20+nn,ndc_entry=1,
                        ndc_mid=19,ndc_nn=nn,ndc_nn_jump=nn,ndc_nn_nojump=0,
                        paired_baseline_recall=.9,paired_baseline_ndc=20,
                        recall=.9 if name=='baseline' else .96,latency_us=100,
                        twohop_mid_scans=int(nn>0),gate_nn_pool_targets=10 if nn else 0,
                        gate_nn_selected_targets=nn,gate_nn_cheap_scores=10 if 'cheap' in name else 0,
                        gate_nn_cheap_products=160 if 'cheap' in name else 0,
                        gate_nn_shadow_scores=0,gate_nn_query_dims_inspected=512 if 'cheap' in name else 0)
                    rows.append(row)
                pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
            (out/'dabs').mkdir()
            pd.DataFrame([dict(row_type='QUERY',query_id=q,ef=0,dabs_gamma=.004,recall=.9699999999999998,
                ndc=25,ndc_entry=1,ndc_mid=24,ndc_nn=0,latency_us=100) for q in (1000,1001)]).to_csv(out/'dabs/dabs.csv',index=False)
            (out/'manifest.json').write_text(json.dumps(manifest))
            command=[sys.executable,str(ROOT/'analysis/analyze_jump_nn_selection.py'),'--indir',str(out)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=pd.read_csv(out/'analysis/online_summary.csv').set_index('config')
            self.assertAlmostEqual(summary.loc['nn_cheap8_d16','distance_equivalent_products'],28+160/512)
            dominated=pd.read_csv(out/'analysis/baseline_dominance.csv')
            self.assertTrue(dominated[dominated.control.eq('dabs')].dominated.all())
            targets=pd.read_csv(out/'analysis/at_target_recall.csv')
            boundary=targets[targets.target_recall.eq(.97)&targets.config.eq('dabs_gamma0.004')].iloc[0]
            self.assertTrue(boundary.reached)
            self.assertEqual(boundary.ndc,25)
            self.assertFalse(targets[targets.target_recall.eq(.97)&targets.config.eq('nn_cheap8_d16')].iloc[0].reached)
            # Reuse synthetic full NDC=10, then verify the all-scored trace
            # reconciliation and conditional rank analysis, including uint64 keys.
            manifest['suite']='nn_diagnostic'; manifest['cases']=[dict(name='nn_full')]
            (out/'manifest.json').write_text(json.dumps(manifest))
            traces=[]
            for qid in (1000,1001):
                for j in range(10):
                    traces.append(dict(query_id=qid,ef=100,probe_id=1,hop=1,nn_id=j,
                        pool_size=10,scored_size=10,censored=0,scored=1,adjacency_ordinal=j,
                        exact_dist=9-j,cheap_score=9-j,random_key=2**64-1-j,
                        first_type=1,second_type=j%3,direct_new_gt=int(j==9),nearest_recovered_desc_gt=0,value_label=int(j==9)))
            pd.DataFrame(traces).to_csv(out/'nn_full_nn_candidates.csv',index=False)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            retention=pd.read_csv(out/'analysis/offline_nn_retention.csv').set_index('selector')
            self.assertEqual(retention.loc['cheap','value_retention'],1)
            self.assertEqual(retention.loc['prefix','value_retention'],0)

    def test_retention_and_censoring(self):
        rows=[]
        for probe,censored in [(1,0),(2,1)]:
            for j in range(10):
                rows.append(dict(query_id=1000,ef=100,probe_id=probe,nn_id=j,
                    pool_size=10,scored_size=10 if not censored else 9,censored=censored,
                    scored=int(not censored or j<9),first_type=1,hop=3,adjacency_ordinal=j,
                    cheap_score=9-j,random_key=9-j,exact_dist=9-j,
                    value_label=int(j==9) if not censored or j<9 else -1,
                    direct_new_gt=int(j==9),nearest_recovered_desc_gt=0))
        result,census=diagnose(pd.DataFrame(rows),8)
        self.assertEqual(census.complete.tolist(),[True,False])
        s=retention_summary(result).set_index('selector')
        self.assertEqual(s.loc['prefix','value_retention'],0)
        self.assertEqual(s.loc['cheap','value_retention'],1)
        self.assertEqual(s.loc['random','value_retention'],1)
        self.assertEqual(s.loc['cheap','exact_fraction'],.8)

    def test_plan(self):
        # Use a workspace temp directory; no dependency on sandbox system temp.
        folder=ROOT/'output/nn_selection_20261008/tests'; folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            out=Path(tmp)/'run'
            command=[sys.executable,str(ROOT/'analysis/run_jump_nn_selection.py'),'--suite','online',
                     '--bin','binary','--query','q','--gt','gt','--index','idx','--out',str(out),'--dry-run','--include-dabs']
            subprocess.run(command,check=True,capture_output=True,text=True)
            m=json.loads((out/'manifest.json').read_text())
            self.assertEqual([r['name'] for r in m['cases']],
                ['baseline','nn_full','nn_prefix8','nn_random8_s17','nn_cheap8_d16','dabs'])
            cheap=m['cases'][4]['command']
            self.assertIn('cheap',cheap); self.assertNotIn('--nn_trace',cheap)
            self.assertEqual(cheap[cheap.index('--jump_max_nn_per_query')+1],'4000')

    def test_parser_names_have_enum_members(self):
        source=(ROOT/'ngfixlib/graph/hnsw_ngfix.h').read_text(encoding='utf-8')
        self.assertNotIn('SearchEdgeType::BASE',source)
        marker='if(partial_nn && !chosen[j-nn_st]) continue;'
        self.assertIn(marker,source)
        self.assertLess(source.index(marker),source.index('q->set_visited(nn);',source.index(marker)))

if __name__=='__main__': unittest.main()
