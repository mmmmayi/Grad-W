#!/bin/bash
#SBATCH -o transCov_2s_lr0.0001_sig10_sig_proty_promse_4n4m_aug0.6.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH --cpus-per-task=16
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

