source /home/mingyuan/lab/subnet/bin/activate
nvidia-smi

cd /work/smile/wanghuan/mz/lit-llama
python finetune/global_subnet.py --data_dir data/${TASK} --sparsity ${SPAR} --ratio ${RATIO}
