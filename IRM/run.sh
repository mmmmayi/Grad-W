#!/bin/bash
#SBATCH -o dysnr_enhanceDisto1e8orm_BLSTM_frameF_0.001_adadelta_4s_wada0.1_th0.1.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa8
####SBATCH -p new
python train.py

