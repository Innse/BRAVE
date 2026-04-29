exp_name=my_experiment
mode="lora"
h5_root=/path/to/h5_patches/
path_to_json=/path/to/pretrain_index.json
output_dir=/path/to/output/${exp_name}_${mode}
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
path_to_log=${output_dir}/dino_${mode}.log

echo "Experiment Name: $exp_name"
echo "H5 Root: $h5_root"
echo "Path to JSON: $path_to_json"
echo "Output Directory: $output_dir"
echo "Log Path: $path_to_log"
echo "Mode: $mode"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main_dino.py \
                                              --wandb_proj_name my_project \
                                              --wandb_exp_name ${exp_name}_${mode} \
                                              --mode ${mode} \
                                              --arch virchow2 \
                                              --h5_root $h5_root \
                                              --data_path $path_to_json \
                                              --output_dir $output_dir \
                                              --saveckp_freq 1 \
                                              --batch_size_per_gpu 48 \
                                              --patch_size 14 \
                                              --local_crops_size 98 > ${path_to_log} 2>&1 &