#!/bin/bash
#SBATCH -o resnet_0.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa7
#SBATCH --cpus-per-task=8
####SBATCH -p new
python train.py

