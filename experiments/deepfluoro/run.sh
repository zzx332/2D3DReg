#!/bin/bash

# 如果你需要激活 conda（推荐）
# source /data/zhouzhexin/code/enter/etc/profile.d/conda.sh
# conda activate polypose

SUBJECT_IDS=(1 2 3 4 5 6)
MODEL_NAME="polypose"

echo "Started at: $(date)"

for SUBJECT_ID in "${SUBJECT_IDS[@]}"
do
    echo "-----------------------------------"
    echo "Running subject: $SUBJECT_ID"
    echo "Model: $MODEL_NAME"
    echo "Started: $(date)"

    python run.py --subject_id $SUBJECT_ID --model_name $MODEL_NAME

    echo "Finished subject: $SUBJECT_ID"
done

echo "All jobs finished at: $(date)"