#!/bin/bash
#SBATCH -o mse_pre0.01_remove0.05_v2_lr0.0005.out
#SBATCH --gres=gpu:2
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

