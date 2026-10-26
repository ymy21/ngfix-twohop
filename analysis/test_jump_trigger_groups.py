import unittest
import numpy as np
import pandas as pd
from analyze_jump_trigger_groups import group_metrics


class TriggerGroupTests(unittest.TestCase):
    def setUp(self):
        self.b=pd.DataFrame({'recall':[.7,.9,1.],'ndc':[3000,3500,4000],'ndc_nn':[0,0,0]})
        self.a=pd.DataFrame({'recall':[.8,.9,1.],'ndc':[6000,7000,8000],'ndc_nn':[4000,4000,4000]})
        self.v=pd.DataFrame({'recall':[.8,.9,1.],'ndc':[6000,3500,4000],'ndc_nn':[4000,0,0]})
    def test_saving_without_losing_helpful_gain(self):
        r=group_metrics(self.v,self.a,self.b)
        self.assertEqual(r['low_queries'],2)
        self.assertEqual(r['helpful_queries'],1)
        self.assertEqual(r['low_nn_saved'],4000)
        self.assertEqual(r['ndc_delta'],-2500)
        self.assertAlmostEqual(r['helpful_net_gain_retention'],1)
        self.assertEqual(r['zero_twohop_queries'],2)
    def test_harm_is_not_clipped_away(self):
        self.v.loc[0,'recall']=.6
        r=group_metrics(self.v,self.a,self.b)
        self.assertAlmostEqual(r['helpful_net_gain_retention'],-1)
        self.assertEqual(r['helpful_lower_than_baseline'],1)
    def test_missing_helpful_group_is_unavailable(self):
        r=group_metrics(self.b,self.b,self.b)
        self.assertTrue(np.isnan(r['helpful_net_gain_retention']))
        self.assertEqual(r['helpful_queries'],0)
    def test_mismatched_queries_are_rejected(self):
        with self.assertRaises(ValueError):group_metrics(self.v.iloc[:2],self.a,self.b)


if __name__=='__main__':unittest.main()
