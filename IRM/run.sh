#!/bin/bash
#SBATCH -o mse_pos_adam_lr0.01_w0.95_hn0.92_n0.001_s8_th0.05_all.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

