#!/bin/bash
#SBATCH -o transCov_twin_30s_lr0.0009_th0.05_p0.5_aug_all.out
#SBATCH --gres=gpu:1
#SBATCH -w hlt06
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

