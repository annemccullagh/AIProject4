# Task 7: Result Analysis

## Which model performed best?
Best model by holdout F1 score: **XGBoost** (Accuracy=0.818, Precision=0.842, Recall=0.941, F1=0.889, ROC-AUC=0.729).

## Which features were most important?
Feature importance method used: **native_feature_importance** from the best model.
- oura_OURA_sleep_onset_latency: 0.058064
- oura_OURA_sleep_score_alignment: 0.037135
- oura_OURA_sleep_rem: 0.034235
- oura_OURA_sleep_restless: 0.031179
- oura_OURA_sleep_average_breath_variation: 0.031076
- oura_OURA_activity_low: 0.029675
- oura_OURA_activity_score_training_volume: 0.028986
- oura_OURA_readiness_score_previous_night: 0.027292
- oura_OURA_sleep_score_total: 0.026090
- oura_OURA_sleep_bedtime_end_delta: 0.025071

## What insights can be drawn about behavior and mental health?
- oura_OURA_sleep_onset_latency is lower on average in high-stress days (low=785.417, high=696.163, delta=-89.254).
- oura_OURA_sleep_score_alignment is higher on average in high-stress days (low=48.083, high=61.023, delta=12.940).
- oura_OURA_sleep_rem is higher on average in high-stress days (low=4466.875, high=5454.070, delta=987.195).
- oura_OURA_sleep_restless is lower on average in high-stress days (low=12.208, high=6.413, delta=-5.796).
- oura_OURA_sleep_average_breath_variation is lower on average in high-stress days (low=3.568, high=3.112, delta=-0.456).
- oura_OURA_activity_low is lower on average in high-stress days (low=2150.500, high=521.953, delta=-1628.547).
- oura_OURA_activity_score_training_volume is lower on average in high-stress days (low=88.167, high=82.709, delta=-5.457).
- oura_OURA_readiness_score_previous_night is higher on average in high-stress days (low=65.833, high=76.116, delta=10.283).
- oura_OURA_sleep_score_total is higher on average in high-stress days (low=70.458, high=73.663, delta=3.204).
- oura_OURA_sleep_bedtime_end_delta is lower on average in high-stress days (low=32876.104, high=29995.744, delta=-2880.360).