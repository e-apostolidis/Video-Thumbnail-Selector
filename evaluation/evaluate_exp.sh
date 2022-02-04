#!/bin/bash

# list of arguments (example of use: bash evaluate.sh <base_path> exp1 OVP)
base_path=$1
exp_id=$2
dataset_name=$3

exp_path="$base_path/$exp_id/$dataset_name"

echo "Extract training data from log files"
for i in $(seq 0 1 9); do
	path="$exp_path/logs/split$i"
	python exportTensorFlowLog.py "$path" "$path"
done

echo "Evaluation using the top-3 selected user thumbnails"
python compute_P@k.py "$exp_path" "$dataset_name"

echo "Evaluation using the top-1 selected user thumbnail"
python compute_P@k_on1thumb.py "$exp_path" "$dataset_name"
