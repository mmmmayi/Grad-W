#!/bin/bash
#SBATCH -o transCov_sclean_0.0001_overall_detachC_noLM.out
#SBATCH --gres=gpu:2
#SBATCH -w ttnusa7
#SBATCH --cpus-per-task=8
#####SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

