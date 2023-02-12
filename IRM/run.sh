#!/bin/bash
#SBATCH -o transCov_msepos_8s_lr0.3_w0.95_hn0.92_n0.001_s8_th0.05_L_eluin.out
#SBATCH --gres=gpu:4
#SBATCH -w ttnusa1
#SBATCH --cpus-per-task=16
#####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py

