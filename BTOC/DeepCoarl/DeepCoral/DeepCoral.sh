#!/usr/bin/env bash
GPU_ID=3
data_dir=/home/xiaoxie/data/tzc/domain_adaptation/DA/data

# Office31
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepCoral/DeepCoral.yaml --data_dir $data_dir --src_domain B --tgt_domain C | tee DeepCoral_D2A.log


