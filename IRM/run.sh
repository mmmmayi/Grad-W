#!/bin/bash
#SBATCH -o saliencyMask_selector_preserve0.1_remove0.01_enh.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
python train.py

