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

## Set up project dir
PROJECT_DIR = "exp/saliencyMask"

## Config
configs = {
    "input_dim": 1793,
    "hidden_units": 128,
    "output_dim": 257,
    "num_layers": 3,        
    "num_epochs": 100,
    "batchsize": 64,
    "data": 'noisy',
    "dur": 4,
    "weight": 0,
    "resume_epoch":None,
    "ratio":0.1,
    "optimizer": {
        "lr": 0.001,
        "beta1": 0.0,
        "beta2": 0.9}}

if __name__ == "__main__":
    ## Dataloader
    # train
    train_irm_dataset = IRMDataset(
        path="../lst/utt_len",
        path_sorted = "../lst/snr_sorted",
        sub=0.1,
        spk="../lst/train_spk.lst",
        batch_size=configs["batchsize"], dur=configs["dur"],
        sampling_rate=16000, mode="train", max_size=200000, data=configs["data"])
    train_loader = DataLoader(
        dataset=train_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=16)
       
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
        num_workers=1)
    ## Model, loss_fn, opt
    if configs['resume_epoch'] is not None:
        nnet = torch.load(f"{PROJECT_DIR}/models/model_{configs['resume_epoch']}.pt")
    else:
        nnet = multi_TDNN()
    #print('Learnable parameters of the model:')
    #for name, param in nnet.named_parameters():
        #if param.requires_grad:
            #print(name, param.numel())
    total_params = sum(p.numel() for p in nnet.parameters() if p.requires_grad)
    print('Number of trainable parameters: {}'.format(total_params))
    #optimizer = torch.optim.Adadelta(nnet.parameters(), lr=configs["optimizer"]["lr"])
    optimizer = torch.optim.Adam(nnet.decoder.parameters())
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
        model=nnet, optimizer=optimizer, loss_fn=[COS_loss,MSE_loss,BCE_loss], dur = configs["dur"],
        train_dl=train_loader, validation_dl=valid_loader, mode=configs["data"],ratio=configs["ratio"])
    #irm_trainer._get_global_mean_variance()
    irm_trainer.train(configs["weight"], configs["resume_epoch"])
