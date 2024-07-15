#!/bin/bash

export TRANSFORMERS_CACHE=/sensei-fs/users/yizhouw/cache/
export HF_HOME=/sensei-fs/users/yizhouw/cache/
export DEEPSPEED_HOME=/sensei-fs/users/yizhouw/cache/
export XDG_CACHE_HOME=/sensei-fs/users/yizhouw/cache/
export TRITON_CACHE_DIR=/sensei-fs/users/yizhouw/cache/

source /home/mingyuan/lab/subnet/bin/activate
nvidia-smi

cd /work/smile/wanghuan/mz/lit-llama
python finetune/global_subnet.py --data_dir data/${TASK} --sparsity ${SPAR} --ratio ${RATIO}
