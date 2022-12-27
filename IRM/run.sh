#!/bin/bash
#SBATCH -o mse100_cw_preserve5_remove0.001_enh1e3_diff100_thf5_pos.out
#SBATCH --gres=gpu:3
#SBATCH -w ttnusa6
#SBATCH --cpus-per-task=24
#####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=3 train.py

