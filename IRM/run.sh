#!/bin/bash
#SBATCH -o transCov_2s_lr0.0001_sig10_sig_ncen_ntest_promse_4n2m.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa3
#SBATCH --cpus-per-task=8
####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

