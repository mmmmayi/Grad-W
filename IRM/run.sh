#!/bin/bash
#SBATCH -o mse_cw_preserve5_remove0.001_enh100.out
#SBATCH --gres=gpu:3
#SBATCH -w ttnusa3
#SBATCH --cpus-per-task=16
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py

