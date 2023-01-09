#!/bin/bash
#SBATCH -o mse_ddp_cos_lr0.0001.out
#SBATCH --gres=gpu:3
#SBATCH -w ttnusa7
#SBATCH --cpus-per-task=16
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py

