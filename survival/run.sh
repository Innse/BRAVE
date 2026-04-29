gpu_id=0
model="ABMIL"
task="Specific_Task"
feature="Specific_Feature"
seed=1 # 2,3 (3 runs)
excel_file="path/to/excel_file.xlsx"
wandb_proj_name="Survival_Analysis_Project"
pt_roots='{"DATASET_A":"/path/to/DATASET_A/pt_files/specific_feature/"}'
h5_roots='{"DATASET_A":"/path/to/DATASET_A/h5_files/"}'


echo "$(date): [task: ${task}]-[feature: ${feature}]-[model: ${model}]-[seed: ${seed}]"
CUDA_VISIBLE_DEVICES=$gpu_id python main_kfold.py --model $model \
                                                                --study $task \
                                                                --feature $feature \
                                                                --excel_file $excel_file \
                                                                --pt_roots $pt_roots \
                                                                --h5_roots $h5_roots \
                                                                --num_epoch 30 \
                                                                --folds 5 \
                                                                --batch_size 1 \
                                                                --seed $seed \
                                                                --wandb_proj_name ${wandb_proj_name} \
                                                                --wandb_exp_name ${task}-${feature}-s${seed}