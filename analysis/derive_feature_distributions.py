"""Exact overall/class quantiles for the four derived distance features."""
import json
import numpy as np
import pandas as pd
from build_complete_experiment_report import MECH,OUT,TABLES,case_info,stats_of,save_json


def main():
    global_rows=[];class_rows=[]
    for folder in sorted(MECH.glob('*_k*/ef*')):
        info=case_info(folder);print('Derived distributions',info['case'],flush=True)
        cols=['mid_dist','cur_dist','bound_at_reject','direct_new_gt','nearest_recovered_desc_gt','admitted_nn','fresh_nn_cmps']
        m=pd.read_csv(folder/'trace_jump_mids.csv',usecols=cols)
        classes=np.where(m.admitted_nn==0,np.where(m.fresh_nn_cmps==0,'FAILED_NO_FRESH','FAILED_WITH_FRESH'),
            np.where((m.direct_new_gt+m.nearest_recovered_desc_gt)>0,'SUCCESS_RECOVERED_GT','SUCCESS_NO_RECOVERED_GT'))
        scale=np.maximum(np.abs(m.bound_at_reject),1e-6)
        s=pd.read_csv(folder/'trace_probe_state.csv',usecols=['topk_q25','topk_q75'])
        if len(s)!=len(m):raise ValueError('State/probe mismatch')
        values={'mid_gap_normalized':(m.mid_dist-m.bound_at_reject)/scale,
                'cur_slack_normalized':(m.bound_at_reject-m.cur_dist)/scale,
                'topk_spread':s.topk_q75-s.topk_q25}
        for field,v in values.items():
            v=v.to_numpy();global_rows.append(dict(**info,file='derived_probe_features',field=field,**stats_of(v)))
            for cls in np.unique(classes):class_rows.append(dict(**info,file='derived_probe_features',field=field,group=cls,**stats_of(v[classes==cls])))
        del m,s,values,scale
        e=pd.read_csv(folder/'trace_jump_events.csv',usecols=['bound_before_nn_admit','nn_dist','jump_class'])
        v=((e.bound_before_nn_admit-e.nn_dist)/np.maximum(np.abs(e.bound_before_nn_admit),1e-6)).to_numpy()
        global_rows.append(dict(**info,file='derived_jump_features',field='nn_margin_normalized',**stats_of(v)))
        for cls in e.jump_class.unique():class_rows.append(dict(**info,file='derived_jump_features',field='nn_margin_normalized',group=cls,**stats_of(v[e.jump_class.to_numpy()==cls])))
        del e,v
    for file,addition in [('all_field_distributions',global_rows),('all_class_field_distributions',class_rows)]:
        existing=pd.read_csv(TABLES/(file+'.csv'))
        existing=existing[~existing.file.str.startswith('derived_')]
        pd.concat([existing,pd.DataFrame(addition)],ignore_index=True).to_csv(TABLES/(file+'.csv'),index=False,encoding='utf-8-sig')
    save_json(OUT/'derived_distribution_manifest.json',dict(global_distributions=len(global_rows),class_distributions=len(class_rows),exact=True,
        derivation='same formulas as analyze_bound_features.py on recorded CSV precision; no source file modification'))
    print('Derived distributions complete',len(global_rows),len(class_rows),flush=True)


if __name__=='__main__':main()
