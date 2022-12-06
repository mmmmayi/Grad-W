#!/bin/bash
#SBATCH -o saliencyMask_selector_preserve0.001.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa8
#SBATCH --cpus-per-task=8
#####SBATCH -p new
python train.py

