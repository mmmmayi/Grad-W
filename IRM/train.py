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
from scheduler import ExponentialDecrease
## Set up project dir
PROJECT_DIR = "exp/transCov_2s_lr0.0001_sig10_sig_ncen_ntest_promse_4n5m_nocons"

## Config
configs = {
    "weight": 1,
    "num_spk": 5,
    "w_n": 0.01,
    "scale":8,  
    "num_epochs": 500,
    "th": 0.05,
    "batchsize": 32,
    "center_num": '4',
    "test_num":'4',
    "dur": 2,
    "resume_epoch":None,
    "ratio":0.1,
    "gpu":[0],
    "optimizer": {
        "initial_lr": 0.0001,
        "final_lr":0.0001,
        "beta1": 0.0,
        "beta2": 0.9}}

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    gpu = int(configs['gpu'][local_rank])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TORCH_DISTRIBUTED_DEBUG']='DETAIL'

    dist.init_process_group(backend="nccl")
    #dist.barrier() 
    ## Dataloader
    # train
    train_irm_dataset = IRMDataset(
        path="../lst/cat_utt_len",
        path_sorted = None,
        sub=1,
        spk="../lst/train_spk.lst",
        batch_size=configs["batchsize"], dur=configs["dur"],mode="train",center_num=configs["center_num"], test_num=configs["test_num"],num_spk=configs["num_spk"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_irm_dataset)
    train_loader = DataLoader(
        dataset=train_irm_dataset,
        batch_size=1,
        shuffle=None,
        sampler=train_sampler,
        num_workers=8)
       
    print('check dataset length:',len(train_irm_dataset))    
    # valid
    valid_irm_dataset = IRMDataset(
        path="../lst/cat_utt_len",
        path_sorted = None,
        sub=1,
        spk="../lst/val_spk.lst",
        batch_size=2,dur=configs["dur"], mode="validation",center_num=configs["center_num"], test_num=configs["test_num"],num_spk=configs["num_spk"])
    valid_loader = DataLoader(
        dataset=valid_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=8)
    world_size = int(os.environ['WORLD_SIZE'])
    print('world_size:',world_size)
    ## Model, loss_fn, opt
    if configs['resume_epoch'] is not None:
        nnet = multi_TDNN(dur=configs["dur"])
        checkpoint = torch.load(f"{PROJECT_DIR}/models/model_{configs['resume_epoch']}.pt")
        nnet.load_state_dict(checkpoint)
    else:
        nnet = multi_TDNN(dur=configs["dur"], scale=configs["scale"])
    print(nnet)
    nnet.cuda()
    #nnet_ddp = nnet.to(f'cuda:{local_rank}')
    nnet_ddp = torch.nn.parallel.DistributedDataParallel(nnet, find_unused_parameters=False)
    #nnet = torch.nn.parallel.DistributedDataParallel(nnet.cuda(local_rank), device_ids=[local_rank], output_device=local_rank)
    #print('Learnable parameters of the model:')
    #for name, param in nnet.named_parameters():
        #if param.requires_grad:
            #print(name, param.numel())
    total_params = sum(p.numel() for p in nnet_ddp.parameters() if p.requires_grad)
    if local_rank == 0:

        print('Number of trainable parameters: {}'.format(total_params))
    #optimizer = torch.optim.Adam(nnet.decoder.parameters(), lr=configs["optimizer"]["lr"])
    optimizer = torch.optim.Adam(list(nnet_ddp.module.decoder.parameters()),lr=configs["optimizer"]["initial_lr"])
    loader_size = len(train_irm_dataset)//world_size
    print(loader_size)
    scheduler = ExponentialDecrease(optimizer,
        num_epochs=configs["num_epochs"],
        epoch_iter=loader_size,
        initial_lr=configs["optimizer"]["initial_lr"],
        final_lr=configs["optimizer"]["final_lr"],
        warm_up_epoch=5,
        scale_ratio= 1*world_size*configs["batchsize"]/64, 
        warm_from_zero=True)
    #for name, param in nnet.named_parameters():
        #if param.requires_grad:
            #print(name, param.numel())
    
    #optimizer = torch.optim.Adadelta(nnet.parameters(), lr=configs["optimizer"]["lr"])
    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()
    COS_loss = nn.CosineSimilarity(dim=1)
    CE_loss = nn.CrossEntropyLoss()
    #loss_fn = nn.L1Loss(reduction='sum')
    #scheduler = MultiStepLR(optimizer=optimizer, milestones=[4, 8, 12, 16], gamma=0.5)

    ## Trainer
    irm_trainer = IRMTrainer(
        config=configs,
        project_dir=PROJECT_DIR,
        model=nnet_ddp, optimizer=optimizer, loss_fn=[COS_loss,MSE_loss,CE_loss], dur = configs["dur"],
        train_dl=train_loader, validation_dl=valid_loader, ratio=configs["ratio"],
        local_rank=local_rank, w_p=configs["weight"], w_n=configs["w_n"],center_num=configs["center_num"])
    device = torch.device("cuda")
    dist.barrier()
    #irm_trainer._get_global_mean_variance()
    for epoch in range(1, configs["num_epochs"]+1):
        irm_trainer.set_models_to_train_mode()
#
        irm_trainer.train_epoch(epoch, configs["weight"], local_rank, loader_size, scheduler)
        if local_rank == 0:

            if epoch%2==0:
                state_dict = nnet.state_dict()
                torch.save(state_dict,f"{PROJECT_DIR}/models/model_{epoch}.pt")
                torch.save(optimizer.state_dict(),f"{PROJECT_DIR}/models/opt_{epoch}.pt")
        irm_trainer.set_models_to_eval_mode()
        irm_trainer.validation_epoch(epoch, configs["weight"], local_rank)
        
    #irm_trainer.train(configs["weight"], local_rank, resume_epoch=configs["resume_epoch"])
