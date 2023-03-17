#!/bin/bash
#SBATCH -o transCov_2s_lr0.001_tfcos_sig10_sig_proty.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa10
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

