#!/bin/bash
#SBATCH -o mse_pos_v2_lr0.1_w0.95_neg0.5_s8_th0.05.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

