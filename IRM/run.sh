#!/bin/bash
#SBATCH -o mse_enh10_ddp_cos.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

