#!/bin/bash
# Minimal run script for BRAVE classification.
# Edit the variables below before running: bash run.sh

study="TCGA-SurPost_Breast_PathSubtype"
feature="brave"
seed=0
gpu_id=0

excel_file="excels/TCGA-SurPost_Breast_PathSubtype.xlsx"
log_dir="./logs/${study}"
mkdir -p "$log_dir"
log_file="${log_dir}/${feature}-ABMIL-${seed}.log"

echo "Study:   $study"
echo "Feature: $feature"
echo "Seed:    $seed"
echo "Log:     $log_file"

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    --study "$study" \
    --feature "$feature" \
    --seed "$seed" \
    --all_datasets ./splits/datasets.xlsx \
    --model ABMIL \
    --excel_file "$excel_file" \
    --num_epoch 50 \
    --early_stop 10 \
    --wandb_proj_name my_project \
    --wandb_exp_name "${study}-${feature}-s${seed}" \
    2>&1 | tee "$log_file"