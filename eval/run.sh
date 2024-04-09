#!/bin/bash

#setp 1: generate 5 responses via chatHPC API 
python -u chathpc_eval.py --eval_data ../data/val/eval_ground_truth.txt --trial_name chathpc1

for trial in $(seq 1 4)
do
   python -u chathpc_eval.py --eval_data eval_chathpc${trial}.txt --trial_name chathpc$((trial+1))
done

#step 2: calculate lexical, semantic, and hallucination scores 
python score.py eval_chathpc5.txt

#step 3: calculate our metric 
python get_aggregated_metric.py eval_chathpc5-scores.csv

