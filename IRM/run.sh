#!/bin/bash
#SBATCH -o transCov_2s_lr0.001_sig10_sig_proty_promse_4n3m.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

