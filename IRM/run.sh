#!/bin/bash
#SBATCH -o transCov_sclean_0.001_noLM_trueL_DFLdiffW_softmax_norelu_spatial_LSNR.out
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa11
#SBATCH --cpus-per-task=8
#SBATCH -p new
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR train.py

