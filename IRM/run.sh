#!/bin/bash
#SBATCH -o mse_cw_preserve_remove0.001_enh100_ddp.out
#SBATCH --gres=gpu:3
#SBATCH -w ttnusa7
#SBATCH --cpus-per-task=16
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py

