#!/bin/bash
#SBATCH -o saliencyMask.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
python train.py

