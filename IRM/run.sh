#!/bin/bash
#SBATCH -o transCov_twin_8s_lr0.003_w0.98_hn0.92_n0.1_s8_th0.05_L_in_all.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

