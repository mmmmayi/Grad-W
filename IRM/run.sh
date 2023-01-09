#!/bin/bash
#SBATCH -o mse_enh_ddp_cos.out
#SBATCH --gres=gpu:2
#SBATCH -w ttnusa3
#SBATCH --cpus-per-task=16
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

