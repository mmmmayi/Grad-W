#!/bin/bash
#SBATCH -o saliencyMask_selector_preserve_remove0.01_enh1e3.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=16
#SBATCH -p new
python train.py

