#!/bin/bash
#SBATCH -o transCov_pos_12s_lr0.3_w0.93_hn0.92_n0.001_s8_th0.05_L_eluin_all.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

