#!/bin/bash
#SBATCH -o transCov_SaM_lr0.0001_mse_5s_16.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa2
#SBATCH --cpus-per-task=8
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

