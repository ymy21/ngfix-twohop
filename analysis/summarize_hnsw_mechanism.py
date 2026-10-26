#!/usr/bin/env python3
"""Summarize paired recall/NDC and optional traces; diagnostic timing is ignored."""
import argparse
import csv
import json
from pathlib import Path
from analyze_hnsw_stage1 import write_csv


def rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        yield from csv.DictReader(f)


def ratio(n,d):
    return n/d if d else ''


def paired_summary(path,label):
    groups={};seen=set()
    keys=('recall','paired_baseline_recall','ndc','paired_baseline_ndc','ndc_entry','ndc_mid','ndc_nn',
          'ndc_nn_jump','ndc_nn_nojump','nn_checked','twohop_mid_scans','jump_mids','jump_candidates',
          'recovered_gt_total','recovered_gt_direct_jump','recovered_gt_bridge_descendant',
          'recovered_gt_non_gt_bridge_root','lost_gt_total')
    for r in rows(path):
        if r['row_type']!='QUERY':continue
        if r['mode']!='boundtwohopcmp':raise ValueError('Need boundtwohopcmp rows')
        ef=int(r['ef']);qid=int(r['query_id'])
        if (ef,qid) in seen:raise ValueError('Duplicate query/ef')
        seen.add((ef,qid))
        g=groups.setdefault(ef,dict(count=0,improved=0,worsened=0,non_hh_exam=0,non_hh_jump=0,**{k:0 for k in keys}))
        g['count']+=1
        g['improved']+=float(r['recall'])>float(r['paired_baseline_recall'])+1e-7
        g['worsened']+=float(r['recall'])<float(r['paired_baseline_recall'])-1e-7
        for k in keys:g[k]+=float(r[k])
        for pair in ('HN','HR','NH','NN','NR','RH','RN','RR'):
            g['non_hh_exam']+=int(r['exam_pair_'+pair]);g['non_hh_jump']+=int(r['jump_pair_'+pair])
        if int(r['ndc'])!=int(r['ndc_entry'])+int(r['ndc_mid'])+int(r['ndc_nn']):
            raise ValueError('NDC partition does not add up')
    if not groups:raise ValueError('No paired queries')
    result=[]
    for ef,g in sorted(groups.items()):
        count=g['count'];recovered=g['recovered_gt_total'];direct=g['recovered_gt_direct_jump'];desc=g['recovered_gt_bridge_descendant']
        row=dict(graph_label=label,ef=ef,queries=count)
        row.update({k+'_mean':g[k]/count for k in keys})
        row.update(delta_recall=(g['recall']-g['paired_baseline_recall'])/count,
                   delta_ndc=(g['ndc']-g['paired_baseline_ndc'])/count,
                   improved_fraction=g['improved']/count,worsened_fraction=g['worsened']/count,
                   net_recovered_gt_mean=(recovered-g['lost_gt_total'])/count,
                   direct_recovered_share=ratio(direct,recovered),routing_recovered_share=ratio(desc,recovered),
                   non_gt_router_recovered_share=ratio(g['recovered_gt_non_gt_bridge_root'],recovered),
                   unattributed_recovered_share=ratio(recovered-direct-desc,recovered),
                   successful_probe_rate=ratio(g['jump_mids'],g['twohop_mid_scans']),
                   admitted_per_fresh_nn=ratio(g['jump_candidates'],g['ndc_nn']),
                   duplicate_or_visited_scan_fraction=ratio(g['nn_checked']-g['ndc_nn'],g['nn_checked']),
                   no_admission_nn_cost_share=ratio(g['ndc_nn_nojump'],g['ndc_nn']),
                   net_gt_per_1k_nn_ndc=ratio(1000*(recovered-g['lost_gt_total']),g['ndc_nn']),
                   non_hh_examined=g['non_hh_exam'],non_hh_admitted=g['non_hh_jump'])
        if label=='unrepaired_bottom' and (g['non_hh_exam'] or g['non_hh_jump']):
            raise ValueError('Repair edge contamination in base mechanism run')
        result.append(row)
    return result


def event_summary(path):
    groups={}
    for r in rows(path):
        visibility=('unseen' if not int(r['baseline_scored_nn']) else
                    'scored_not_expanded' if not int(r['baseline_expanded_nn']) else 'expanded')
        key=(int(r['ef']),r['jump_class'],visibility)
        g=groups.setdefault(key,dict(events=0,nearest_recovered_descendants=0,query_ids=set()))
        g['events']+=1;g['nearest_recovered_descendants']+=int(r['nearest_recovered_gt_descendants'])
        g['query_ids'].add(int(r['query_id']))
    return [dict(ef=k[0],jump_class=k[1],baseline_nn_visibility=k[2],events=g['events'],
                 queries_with_event=len(g['query_ids']),nearest_recovered_descendants=g['nearest_recovered_descendants'])
            for k,g in sorted(groups.items())]


def edge_pair_summary(path):
    groups={}
    for r in rows(path):
        if r['row_type']!='QUERY':continue
        for pair in ('HH','HN','HR','NH','NN','NR','RH','RN','RR'):
            key=(int(r['ef']),pair)
            g=groups.setdefault(key,dict(examined=0,admitted=0,fresh_scored=0,has_scores=True))
            g['examined']+=int(r['exam_pair_'+pair]);g['admitted']+=int(r['jump_pair_'+pair])
            if 'score_pair_'+pair in r:g['fresh_scored']+=int(r['score_pair_'+pair])
            else:g['has_scores']=False
    return [dict(ef=k[0],edge_pair=k[1],examined=g['examined'],admitted=g['admitted'],
                 fresh_scored=g['fresh_scored'] if g['has_scores'] else '',
                 admitted_per_fresh_scored=ratio(g['admitted'],g['fresh_scored']) if g['has_scores'] else '')
            for k,g in sorted(groups.items())]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--events',type=Path)
    p.add_argument('--graph-label',choices=('unrepaired_bottom','repaired_reference'),default='unrepaired_bottom')
    p.add_argument('--clean-dir',type=Path,help='Require paired diagnostic Recall/NDC to match the completed clean sweep')
    p.add_argument('--ngfix-paired',type=Path,help='Historical diagnostics only; user must verify matching data/K/profile')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.clean_dir:
        manifest=json.loads((a.clean_dir/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('suite')!='hnsw_stage1' or not manifest.get('completed'):
            raise ValueError('Need a completed clean sweep for cross-checking')
        if manifest['graph']!=a.graph_label:
            raise ValueError('Graph label differs from clean sweep')
        lookup={(r['method'],int(r['ef']),int(r['query_id'])):r
                for r in rows(a.clean_dir/'timings.csv.queries.csv')}
        for r in rows(a.paired):
            if r['row_type']!='QUERY':continue
            for method,recall_key,ndc_key in [('beam','paired_baseline_recall','paired_baseline_ndc'),
                                             ('bound_unbounded','recall','ndc')]:
                key=(method,int(r['ef']),int(r['query_id']))
                if key not in lookup:
                    raise ValueError('Diagnostic query/ef outside clean sweep')
                clean=lookup[key]
                if abs(float(clean['recall'])-float(r[recall_key]))>1e-6 or int(clean['ndc'])!=int(r[ndc_key]):
                    raise ValueError('Diagnostic/clean result mismatch at '+str(key))
    base=paired_summary(a.paired,a.graph_label)
    write_csv(a.out/'paired_mechanism_summary.csv',base)
    write_csv(a.out/'twohop_edge_pair_summary.csv',edge_pair_summary(a.paired))
    if a.events:
        # Empty event files are legitimate: a run may admit no jumps.
        write_csv(a.out/'jump_class_by_baseline_visibility.csv',event_summary(a.events))
    if a.ngfix_paired:
        prior=paired_summary(a.ngfix_paired,'historical_ngfix_unchecked_metadata')
        write_csv(a.out/'historical_ngfix_summary.csv',prior)
        # Compare only exactly shared query IDs at each ef. Still descriptive
        # until metric/K/query hashes/index/profile/version have been audited.
        left={(int(r['ef']),int(r['query_id'])):r for r in rows(a.paired) if r['row_type']=='QUERY'}
        right={(int(r['ef']),int(r['query_id'])):r for r in rows(a.ngfix_paired) if r['row_type']=='QUERY'}
        common=left.keys() & right.keys();groups={}
        for key in sorted(common):
            g=groups.setdefault(key[0],dict(ef=key[0],shared_queries=0,base_twohop_recall=0,ngfix_twohop_recall=0,
                                           base_baseline_recall=0,ngfix_baseline_recall=0))
            g['shared_queries']+=1
            for label,source in [('base',left[key]),('ngfix',right[key])]:
                g[label+'_twohop_recall']+=float(source['recall'])
                g[label+'_baseline_recall']+=float(source['paired_baseline_recall'])
        for g in groups.values():
            for key in ('base_twohop_recall','ngfix_twohop_recall','base_baseline_recall','ngfix_baseline_recall'):
                g[key]/=g['shared_queries']
        write_csv(a.out/'historical_shared_queries_descriptive.csv',list(groups.values()))
    print('Mechanism summaries (no diagnostic QPS used):',a.out)


if __name__=='__main__':main()
