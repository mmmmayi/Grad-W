#!/bin/bash
#SBATCH -o speaker.out
#SBATCH --gres=gpu:1
####SBATCH -w ttnusa2
#####SBATCH -p new
python apply.py

