import unittest
import pandas as pd
from analyze_bound_features import joined,add_features,feature_bins,KEY


class BoundFeatureTests(unittest.TestCase):
    def fixture(self):
        mids=pd.DataFrame([
            dict(query_id=0,ef=200,cur_id=1,mid_id=2,hop=4,bound_at_reject=.5,
                 mid_dist=.6,cur_dist=.3,admitted_nn=0,fresh_nn_cmps=10,direct_new_gt=0,nearest_recovered_desc_gt=0),
            dict(query_id=1,ef=200,cur_id=3,mid_id=4,hop=4,bound_at_reject=.5,
                 mid_dist=.52,cur_dist=.3,admitted_nn=2,fresh_nn_cmps=20,direct_new_gt=1,nearest_recovered_desc_gt=2)])
        states=mids[KEY].copy();states['state_ndc']=[100,200]
        states['topk_q25']=.1;states['topk_q75']=.3
        return mids,states

    def test_failed_controls_and_observed_recovered_yield(self):
        mids,states=self.fixture();frame=add_features(joined(mids,states))
        self.assertEqual(frame.probe_class.tolist(),['FAILED_WITH_FRESH','SUCCESS_RECOVERED_GT'])
        self.assertEqual(frame.recovered_gt.tolist(),[0,3])
        table=feature_bins(frame)
        useful=table[(table.feature=='state_ndc') & (table.feature_bin=='200')].iloc[0]
        self.assertAlmostEqual(useful.recovered_gt_per_1k_fresh_nn_ndc,150)

    def test_snapshot_keys_must_match(self):
        mids,states=self.fixture()
        with self.assertRaises(ValueError):joined(mids,states.iloc[:1])
        with self.assertRaises(ValueError):joined(mids,pd.concat([states,states.iloc[:1]]))

    def test_no_fresh_distance_is_a_separate_failure(self):
        mids,states=self.fixture();mids.loc[0,'fresh_nn_cmps']=0
        frame=add_features(joined(mids,states))
        self.assertEqual(frame.iloc[0].probe_class,'FAILED_NO_FRESH')


if __name__=='__main__':unittest.main()
