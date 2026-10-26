#!/usr/bin/env python3
"""Synthetic workflow checks. These fixtures are not ANN measurements."""
import csv
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from run_jump_schedule import build_cases


class ScheduleWorkflowTests(unittest.TestCase):
    def test_search_edge_enum_references_exist(self):
        root=Path(__file__).resolve().parent.parent
        source=(root/'ngfixlib/graph/hnsw_ngfix.h').read_text(encoding='utf-8')
        declaration=re.search(r'enum class SearchEdgeType\s*:\s*\w+\s*\{([^}]+)\}',source)
        self.assertIsNotNone(declaration)
        members=set(re.findall(r'\b(\w+)\s*=',declaration.group(1)))
        references=set(re.findall(r'SearchEdgeType::(\w+)',source))
        self.assertEqual(references-members,set(), 'Undeclared SearchEdgeType member')

    def test_selection_and_trigger_controls(self):
        args=SimpleNamespace(suite='selection',gap=.06,nn_cap=4000,mid_cap=2,
                             nn_per_mid=0,nn_per_cur=0,selection='gap',windows='4,8')
        cases=dict((n,o) for n,_,o in build_cases(args))
        for name in ['select_adjacency','select_gap','select_stratified']:
            options=cases[name]
            self.assertEqual(options[options.index('--jump_max_nn_per_query')+1],'4000')
            self.assertEqual(options[options.index('--jump_schedule')+1],'after_onehop')
        self.assertNotIn('--jump_schedule',cases['inline_adj'])
        self.assertNotIn('--jump_max_mids_per_cur',cases['after_all'])
        args.suite='trigger'
        cases=dict((n,o) for n,_,o in build_cases(args))
        for name,options in cases.items():
            if name.startswith('trigger_'):
                self.assertEqual(options[options.index('--jump_mid_selection')+1],'gap')
                self.assertEqual(options[options.index('--jump_max_mids_per_cur')+1],'2')

    def test_accounting_and_same_query_comparison(self):
        scripts=Path(__file__).resolve().parent
        repository=scripts.parent.resolve()
        fixture_parent=repository/'output/jump_schedule_checks'
        fixture_parent.mkdir(parents=True,exist_ok=True)
        if not fixture_parent.resolve().is_relative_to(repository):
            raise ValueError('Fixture directory must stay inside the repository')
        with tempfile.TemporaryDirectory(prefix='jump_schedule_fixture_',dir=fixture_parent) as tmp:
            root=Path(tmp)
            names=['baseline','original','legacy_budget','select_adjacency','select_gap','select_stratified']
            manifest=dict(suite='selection',cases=[dict(name=n) for n in names],
                controls_settings=dict(offset=1000,limit=2),
                parameters=dict(nn_cap=4000,nn_per_mid=0,nn_per_cur=0))
            (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            metrics=['ndc_mid','ndc_nn','ndc_nn_nojump','twohop_mid_scans','jump_mids',
                'jump_candidates','jump_candidates_gt','recovered_gt_total','recovered_gt_direct_jump',
                'recovered_gt_non_gt_bridge_root','lost_gt_total','latency_us','hops']
            gates=['rejected_mids','blocked_hop','blocked_cur','blocked_gap','blocked_edge','blocked_budget',
                'blocked_policy','blocked_detour','blocked_selection','blocked_trigger','blocked_batch_budget',
                'probed_base_mids','probed_ngfix_mids','probed_rfix_mids']
            frames={}
            for name in names:
                rows=[]
                for qid in [1000,1001]:
                    baseline=name=='baseline'
                    row={k:0 for k in metrics}
                    row.update({'gate_'+k:0 for k in gates})
                    row.update(row_type='QUERY',mode='baseline' if baseline else 'boundtwohopcmp',
                        ef=100,query_id=qid,recall=.9 if baseline else .98,
                        ndc=20 if baseline else 26,ndc_entry=1,ndc_mid=19,
                        ndc_nn=0 if baseline else 6,ndc_nn_jump=0 if baseline else 6,
                        paired_baseline_recall=.9,paired_baseline_ndc=20)
                    if not baseline:
                        row.update(twohop_mid_scans=1,gate_rejected_mids=2,
                                   gate_blocked_selection=1,gate_probed_base_mids=1)
                    rows.append(row)
                frames[name]=rows
                with (root/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (root/'dabs').mkdir()
            dabs_rows=[]
            for qid in [1000,1001]:
                row=dict(frames['baseline'][0],query_id=qid,mode='dabs',ef=0,dabs_gamma=.002,
                         recall=.99,ndc=24,ndc_mid=23)
                dabs_rows.append(row)
            with (root/'dabs/dabs.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(dabs_rows[0]));writer.writeheader();writer.writerows(dabs_rows)
            for script in ['summarize_jump_gates.py','compare_jump_schedule.py']:
                subprocess.run([sys.executable,str(scripts/script),'--indir',str(root)],check=True,capture_output=True)
            with (root/'analysis/mechanism_comparison.csv').open() as f:
                comparison=list(csv.DictReader(f))
            self.assertEqual(len(comparison),5)
            with (root/'analysis/combined_targets.csv').open() as f:
                targets=list(csv.DictReader(f))
            dabs_target=next(r for r in targets if r['method']=='dabs' and float(r['target_recall'])==.99)
            self.assertEqual(float(dabs_target['ndc']),24)
            # A missing blocking reason must fail, rather than silently suggest saved work.
            frames['select_gap'][0]['gate_blocked_selection']=0
            with (root/'select_gap.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(frames['select_gap'][0]));writer.writeheader();writer.writerows(frames['select_gap'])
            result=subprocess.run([sys.executable,str(scripts/'summarize_jump_gates.py'),'--indir',str(root)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Gate accounting mismatch',result.stderr)


if __name__=='__main__':
    unittest.main()
