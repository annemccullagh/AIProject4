# Task 7: Result Analysis

## Which model performed best?
Best model by holdout F1 score: **XGBoost** (Accuracy=0.818, Precision=0.842, Recall=0.941, F1=0.889, ROC-AUC=0.741).

## Which features were most important?
Feature importance method used: **native_feature_importance** from the best model.
- oura_OURA_sleep_average_breath_variation: 0.036795
- oura_OURA_sleep_midpoint_at_delta: 0.035987
- oura_OURA_sleep_restless: 0.035358
- oura_OURA_sleep_score_alignment: 0.034602
- oura_OURA_activity_score_training_volume: 0.033801
- oura_OURA_activity_score: 0.033773
- oura_OURA_activity_low: 0.031608
- oura_OURA_sleep_deep: 0.031579
- oura_OURA_activity_cal_total: 0.027990
- oura_OURA_sleep_score_rem: 0.026972

## What insights can be drawn about behavior and mental health?
- oura_OURA_sleep_average_breath_variation is lower on average in high-stress days (low=3.568, high=3.112, delta=-0.456).
- oura_OURA_sleep_midpoint_at_delta is lower on average in high-stress days (low=18357.354, high=15534.698, delta=-2822.656).
- oura_OURA_sleep_restless is lower on average in high-stress days (low=12.208, high=6.413, delta=-5.796).
- oura_OURA_sleep_score_alignment is higher on average in high-stress days (low=48.083, high=61.023, delta=12.940).
- oura_OURA_activity_score_training_volume is lower on average in high-stress days (low=88.167, high=82.709, delta=-5.457).
- oura_OURA_activity_score is lower on average in high-stress days (low=80.458, high=76.221, delta=-4.237).
- oura_OURA_activity_low is lower on average in high-stress days (low=2150.500, high=521.953, delta=-1628.547).
- oura_OURA_sleep_deep is higher on average in high-stress days (low=6659.375, high=7287.035, delta=627.660).
- oura_OURA_activity_cal_total is lower on average in high-stress days (low=2328.417, high=2115.093, delta=-213.324).
- oura_OURA_sleep_score_rem is higher on average in high-stress days (low=64.708, high=75.302, delta=10.594).