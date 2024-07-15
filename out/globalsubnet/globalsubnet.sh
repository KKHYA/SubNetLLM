cd ../..
python finetune/global_subnet.py --data_dir data/${TASK} --sparsity ${SPAR} --ratio ${RATIO}
