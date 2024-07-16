#!/bin/bash

export TRANSFORMERS_CACHE=/sensei-fs/users/yizhouw/cache/
export HF_HOME=/sensei-fs/users/yizhouw/cache/
export DEEPSPEED_HOME=/sensei-fs/users/yizhouw/cache/
export XDG_CACHE_HOME=/sensei-fs/users/yizhouw/cache/
export TRITON_CACHE_DIR=/sensei-fs/users/yizhouw/cache/

cd

echo "\n\n ==> Initialize Conda environment..."

source /sensei-fs/users/yizhouw/init_train.sh

echo "Conda initialization completed."

cd /sensei-fs/users/yizhouw/mingyuan/

source work/smile/wanghuan/mz/subnet/bin/activate

# finish conda initialization

echo "\n\n ==> Test Pytorch version..."
pip show torch

# Set DeepSpeed cache directory
export DS_CACHE_DIR=/sensei-fs/users/yizhouw/deepspeed_cache/
mkdir -p $DS_CACHE_DIR

# begin running experiments


echo "\n\n ==> cd project..."
cd /sensei-fs/users/yizhouw/mingyuan/SubNetLLM

echo "\n\n ==> Start runing experiments..."

python finetune/global_subnet.py --data_dir data/alpaca --sparsity 0.5 --ratio 1.0
