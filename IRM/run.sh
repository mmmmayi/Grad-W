#!/bin/bash
#SBATCH -o saliencyMask_selector_preserve_remove0.01_enh.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa8
#SBATCH --cpus-per-task=8
####SBATCH -p new
python train.py

