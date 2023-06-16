#!/bin/bash
#SBATCH -o transCov_sclean_0.00001_overall_detachC_noLM_encoder_weight_snr_norm_mae.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa12
#SBATCH --cpus-per-task=8
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py

