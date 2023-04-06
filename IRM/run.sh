#!/bin/bash
#SBATCH -o transCov_lr0.0001_sig10_sig_ctest2_fixcen20_promse_4n2m4k.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=8
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

