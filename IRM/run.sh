#!/bin/bash
#SBATCH -o mse_ddp_output_sgd_v2_lr0.0007.out
#SBATCH --gres=gpu:2
#SBATCH -w ttnusa7
#SBATCH --cpus-per-task=16
#####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

