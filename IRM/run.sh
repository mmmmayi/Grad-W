#!/bin/bash
#SBATCH -o mse_pre0.01_remove0.02_v2_lr0.0005.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa2
#SBATCH --cpus-per-task=16
#####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

