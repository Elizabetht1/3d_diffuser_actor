#!/usr/bin/bash

exp_id=$1

if [ -z "$exp_id" ]; then
    echo "Usage: $0 <exp_id>"
    exit 1
fi

weight_dir=test_weights/$exp_id
config=$weight_dir/hparams.json

if [ ! -f "$config" ]; then
    echo "Error: $config not found"
    exit 1
fi

# prefer best.pth, fall back to recent.pth
if [ -f "$weight_dir/recent.pth" ]; then
    checkpoint=$weight_dir/recent.pth
elif [ -f "$weight_dir/best.pth" ]; then
    checkpoint=$weight_dir/best.pth
else
    echo "Error: no checkpoint found in $weight_dir"
    exit 1
fi

# infer from hparams
mapfile -t tasks < <(jq -r '.tasks[]' $config)
cameras=$(jq -r '.views | join(",")' $config)
embed_type=$(jq -r '.embed_type' $config)
device=$(jq -r '.device' $config)

data_dir=/data/rlbench2/val
num_episodes=10
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds_corrected.json
use_instruction=1
max_tries=2
verbose=1
single_task_gripper_loc_bounds=0
seed=0
quaternion_format=wxyz  # IMPORTANT: change this to be the same as the training script IF you're not using our checkpoint
headless=1
image_size=128,128
export COPPELIASIM_ROOT=~/Desktop/projects/3d_diffuser_actor/PyRep/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

echo "exp_id:     $exp_id"
echo "checkpoint: $checkpoint"
echo "config:     $config"
echo "tasks:      ${tasks[*]}"
echo "cameras:    $cameras"
echo "embed_type: $embed_type"
echo "device:     $device"
echo ""

num_tasks=${#tasks[@]}
for ((i=0; i<$num_tasks; i++)); do
    DISPLAY=:1 python -m online_evaluation_rlbench.evaluate_policy \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp_id/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 45 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --quaternion_format $quaternion_format \
    --dense_interpolation 1 \
    --headless $headless \
    --config $config \
    --image_size $image_size \
    --device $device \
    --verify 1 \
    --embed_type $embed_type
done
