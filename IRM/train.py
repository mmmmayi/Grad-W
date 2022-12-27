#!/usr/bin/env python3
import sys, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from Models.dataset import IRMDataset, ToTensor
from Models.models import BLSTM
from Models.TDNN import multi_TDNN
from Trainer.trainer import IRMTrainer
import torch.distributed as dist
## Set up project dir
PROJECT_DIR = "exp/mse100_cw_preserve5_remove0.001_enh1e3_diff100_thf5_pos"

## Config
configs = {
    "input_dim": 1793,
    "hidden_units": 128,
    "output_dim": 257,
    "num_layers": 3,        
    "num_epochs": 100,
    "batchsize": 14,
    "data": 'noisy',
    "dur": 4,
    "weight": 1000,
    "resume_epoch":None,
    "ratio":0.1,
    "gpu":[0,1,2],
    "optimizer": {
        "lr": 0.001,
        "beta1": 0.0,
        "beta2": 0.9}}

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu = int(configs['gpu'][local_rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    dist.init_process_group(backend="nccl")
    #dist.barrier() 
    ## Dataloader
    # train
    train_irm_dataset = IRMDataset(
        path="../lst/utt_len",
        path_sorted = "../lst/snr_sorted",
        sub=0.1,
        spk="../lst/train_spk.lst",
        batch_size=configs["batchsize"], dur=configs["dur"],
        sampling_rate=16000, mode="train", max_size=200000, data=configs["data"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_irm_dataset)
    train_loader = DataLoader(
        dataset=train_irm_dataset,
        batch_size=1,
        shuffle=None,
        sampler=train_sampler,
        num_workers=24)
       
    #print('check dataset length:',len(train_irm_dataset))    
    # valid
    valid_irm_dataset = IRMDataset(
        path="../lst/utt_len",
        path_sorted = "../lst/snr_sorted",
        sub=1,
        spk="../lst/val_spk.lst",
        batch_size=2,dur=5,
        sampling_rate=16000, mode="validation", max_size=100000, data=configs["data"])
    valid_loader = DataLoader(
        dataset=valid_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=8)
    ## Model, loss_fn, opt
    if configs['resume_epoch'] is not None:
        nnet = torch.load(f"{PROJECT_DIR}/models/model_{configs['resume_epoch']}.pt")
    else:
        nnet = multi_TDNN(dur=configs["dur"])
    nnet.cuda()
    nnet_ddp = nnet.to(f'cuda:{local_rank}')
    nnet_ddp = torch.nn.parallel.DistributedDataParallel(nnet_ddp, find_unused_parameters=False)
    #nnet = torch.nn.parallel.DistributedDataParallel(nnet.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    #print('Learnable parameters of the model:')
    #for name, param in nnet.named_parameters():
        #if param.requires_grad:
            #print(name, param.numel())
    total_params = sum(p.numel() for p in nnet_ddp.parameters() if p.requires_grad)
    if local_rank == 0:

        print('Number of trainable parameters: {}'.format(total_params))
    #optimizer = torch.optim.Adam(nnet.decoder.parameters(), lr=configs["optimizer"]["lr"])
    optimizer = torch.optim.Adam(list(nnet_ddp.module.decoder.parameters())+list(nnet_ddp.module.embedding.parameters()))
    #for name, param in nnet.named_parameters():
        #if param.requires_grad:
            #print(name, param.numel())
    
    #optimizer = torch.optim.Adadelta(nnet.parameters(), lr=configs["optimizer"]["lr"])
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()
    COS_loss = nn.CosineEmbeddingLoss()
    CE_loss = nn.CrossEntropyLoss()
    #loss_fn = nn.L1Loss(reduction='sum')
    #scheduler = MultiStepLR(optimizer=optimizer, milestones=[4, 8, 12, 16], gamma=0.5)

    ## Trainer
    irm_trainer = IRMTrainer(
        config=configs,
        project_dir=PROJECT_DIR,
        model=nnet_ddp, optimizer=optimizer, loss_fn=[COS_loss,MSE_loss,BCE_loss], dur = configs["dur"],
        train_dl=train_loader, validation_dl=valid_loader, mode=configs["data"],ratio=configs["ratio"],local_rank=local_rank)
    device = torch.device("cuda")
    dist.barrier()
    #irm_trainer._get_global_mean_variance()
    for epoch in range(1, configs["num_epochs"]+1):
        irm_trainer.set_models_to_train_mode()
        irm_trainer.train_epoch(epoch, configs["weight"], local_rank)
        if local_rank == 0:

            if epoch%1==0:
                state_dict = nnet.state_dict()
                torch.save(state_dict,f"{PROJECT_DIR}/models/model_{epoch}.pt")
        irm_trainer.set_models_to_eval_mode()
        irm_trainer.validation_epoch(epoch, configs["weight"], local_rank)
    #irm_trainer.train(configs["weight"], local_rank, resume_epoch=configs["resume_epoch"])
