#!/usr/bin/env bash
GPU_ID=0
data_dir=/home/xiaoxie/data/tzc/domain_adaptation/DA/data

# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DSAN/DSAN.yaml --data_dir $data_dir --src_domain A --tgt_domain B | tee DSAN_D2A.log



