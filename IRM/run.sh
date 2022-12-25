#!/bin/bash
#SBATCH -o temp.out
#SBATCH --gres=gpu:2
#SBATCH -w ttnusa6
#SBATCH --cpus-per-task=16
#####SBATCH -p new
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

