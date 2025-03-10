#!/bin/bash

dataset=$1
model_path=$2
model_base=$3
cache_dir=$4
data_cache_dir=$5

# Only single GPU is currently supported for evaluation
export CUDA_VISIBLE_DEVICES=0

# Start building the command
cmd="python videollava/eval/eval.py \
    --dataset $dataset \
    --model_path $model_path \
    --load_8bit \
    --prompt_strategy interleave \
    --chronological_prefix"

# Add model_base if provided
if [ ! -z "$model_base" ]; then
    cmd="$cmd --model_base $model_base"
fi

# Add cache_dir if provided
if [ ! -z "$cache_dir" ]; then
    cmd="$cmd --cache_dir $cache_dir"
fi

# Add data_cache_dir if provided
if [ ! -z "$data_cache_dir" ]; then
    cmd="$cmd --data_cache_dir $data_cache_dir"
fi

# Execute the command
eval $cmd
