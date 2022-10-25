import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time
import os, torchaudio
import numpy as np
from Models.models import PreEmphasis
from Models.TDNN import Speaker_TDNN
import tools.utils as utils
from tools.logger import get_logger
from torch.utils.tensorboard import SummaryWriter
from tools.metrics import compute_PESQ, compute_segsnr
import torch.nn.functional as F

class IRMTrainer():
    def __init__(self,
            config, project_dir,
            model, optimizer, loss_fn, dur,
            train_dl, validation_dl, mode, ratio):
        # Init device
        self.config = config

        # Dataloaders
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl  ##to check

        # Model Params
        self.optimizer = optimizer
        self.ratio = ratio
        self.cos = loss_fn[0]
        self.mse = loss_fn[1]
        self.bce = loss_fn[2]
        self.last_loss = float('inf')
        self.time = 0
        self.model = model
        self.mode = mode
        self.dur = dur
        self.pre = PreEmphasis()
        self.Spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2).cuda()
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1).cuda()

        self.MelSpec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80).cuda()
        # Init project dir
        self.PROJECT_DIR = project_dir
        if not os.path.exists(f"{self.PROJECT_DIR}/models"):
            os.makedirs(f"{self.PROJECT_DIR}/models")
        
        self.mean = torch.load("../lst/mean.pt").float().cuda()
        self.std = torch.load("../lst/std.pt").float().cuda()

        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Setup logger
        comment = project_dir.split('/')[-1]
        self.writer = SummaryWriter(comment=comment)
        self.logger = get_logger(f"{self.PROJECT_DIR}/LOG")
        self.logger.info(self.config)
        self.speaker = Speaker_TDNN().cuda()
        self_state = self.speaker.state_dict()
        path = '../exps/pretrain.model'
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            if name not in self_state:
                name = name.replace("speaker_encoder.","")
                if name not in self_state:
                    continue
            self_state[name].copy_(param)
        for p in self.speaker.parameters():
            p.requires_grad = False
        self.speaker.eval()

        print(f"log file at: {self.PROJECT_DIR}/LOG.log")

    def __set_models_to_train_mode(self):
        self.model.train()

    def vari_sigmoid(self,x,a):
        return 1/(1+(-a*x).exp())

    def vari_ReLU(self,x,th):
        assert th>0 and th<1
        relu = nn.ReLU()
        x = relu(x)
        max = torch.amax(x,dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        reg = torch.where(x > max*th, x, torch.tensor(0, dtype=x.dtype).cuda())
        return reg/max

    def _clip_point(self, frame_len):
        clip_length = 100*self.dur+1
        start_point = 0
        end_point = frame_len
        if frame_len>=clip_length:
            start_point = np.random.randint(0, frame_len - clip_length + 1)
            end_point = start_point + clip_length
        else:
            print('bug exists in clip point')
            quit()
        return start_point, end_point


    def __train_epoch(self, epoch, weight):
        start_time = time.time()
        relu = nn.ReLU()
        running_mse, running_cos, running_x3, running_x = 0,0,0,0
        i_batch = 0
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            # Load data from trainloader
            Xb, target_spk, clean, check  = sample_batched
            
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda()
            clean = clean.reshape(B,W).cuda()
            target_spk = target_spk.reshape(B)
            with torch.no_grad():
                Xb = self.pre(Xb)
                Xb = (self.Spec(Xb)+1e-8).log()           
                clean = self.pre(clean)
                clean = (self.Spec(clean)+1e-8).log()     
 
                frame_len = Xb.shape[-1]
                start, end = self._clip_point(frame_len) 
                Xb = Xb[:,:,start:end]
                clean = clean[:,:,start:end]
                mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()
                mel_c = (self.Mel_scale(clean.exp())+1e-6).log()
                feature = mel_c.requires_grad_()

                Xb_input = (Xb-self.mean)/self.std
 
            score = self.speaker(feature, target_spk.cuda(), 'score')
            self.speaker.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            
            self.optimizer.zero_grad()
            
            cam = self.model(mel_n, Xb_input, target_spk)
            pred_spec = (cam+1e-8).log()+Xb
            pred_mel = (self.Mel_scale(pred_spec.exp())+1e-6).log()
            mse_loss = self.mse(pred_spec, clean)/B
            disto_loss = self.vari_ReLU(0-yb,self.ratio)*torch.pow(relu(pred_mel-mel_c),2)+self.vari_ReLU(yb,self.ratio)*torch.pow(relu(mel_c-pred_mel),2)
            disto_loss = torch.mean(disto_loss)

            train_loss = mse_loss+disto_loss*weight
            running_mse += mse_loss.item()
            running_cos += disto_loss.item()
            
            try:
                train_loss.backward()
            except RuntimeError as e:
                print(e)
                quit()
            
            self.optimizer.step()
            torch.cuda.empty_cache()
            i_batch += 1
            self.logger.info("Loss of minibatch {}-th/{}: {}, lr:{}".format(i_batch, len(self.train_dataloader), train_loss.item(), self.optimizer.param_groups[0]['lr']))
        if epoch%10 ==0:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.8
        if epoch%10 ==0:
            torch.save(self.model, f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
        ave_mse_loss = running_mse / i_batch
        ave_cos_loss = running_cos / i_batch
        end_time = time.time()
        self.writer.add_scalar('Train/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Train/cos', ave_cos_loss, epoch)
        self.logger.info("Epoch:{}, loss = {}".format(epoch, ave_mse_loss+ave_cos_loss)) 
        self.logger.info(f"Time used for this epoch training: {end_time - start_time} seconds")
        self.logger.info("*" * 50)
    

    
    def __set_models_to_eval_mode(self):
        self.model.eval()

    def __validation_epoch(self, epoch, weight):
        start_time = time.time()
        running_mse_loss, running_cos_loss, running_x3_loss, running_x_loss= 0.0, 0.0, 0.0, 0.0
        i_batch = 0
        relu = nn.ReLU()

        print("Validating...")
        for sample_batched in tqdm(self.validation_dataloader):
            Xb, target_spk, clean, check= sample_batched
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda()
            clean = clean.reshape(B,W).cuda()
            target_spk = target_spk.squeeze()
           
            with torch.no_grad():
                Xb = self.pre(Xb)
                Xb = (self.Spec(Xb)+1e-8).log()
                clean = self.pre(clean)
                clean = (self.Spec(clean)+1e-8).log()

                frame_len = Xb.shape[-1]
                mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()
                mel_c = (self.Mel_scale(clean.exp())+1e-6).log()
                feature = mel_c.requires_grad_()
                Xb_input = (Xb-self.mean)/self.std

            score = self.speaker(feature, target_spk.cuda(), 'score')
            self.speaker.zero_grad()

            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            cam = self.model(mel_n, Xb_input, target_spk)

            pred_spec = (cam+1e-8).log()+Xb
            pred_mel = (self.Mel_scale(pred_spec.exp())+1e-6).log()
 
            mse_loss = self.mse(pred_spec, clean)/B
            disto_loss = self.vari_ReLU(0-yb,self.ratio)*torch.pow(relu(pred_mel-mel_c),2)+self.vari_ReLU(yb,self.ratio)*torch.pow(relu(mel_n-pred_mel),2)
            disto_loss = torch.mean(disto_loss)
            running_mse_loss += mse_loss.item()
            running_cos_loss += disto_loss.item()

            i_batch += 1
            
        ave_cos_loss = running_cos_loss / i_batch
        ave_mse_loss = running_mse_loss / i_batch
        end_time = time.time()
        self.writer.add_scalar('Validation/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Validation/cos', ave_cos_loss, epoch)
        self.logger.info(f"Time used for this epoch validation: {end_time - start_time} seconds")
        self.logger.info("Epoch:{}, average loss = {}".format(epoch, ave_cos_loss+ave_mse_loss))

    def _get_global_mean_variance(self,mag_type):
        mean = 0.0
        variance = 0.0
        N_ = 0
        for sample_batched in tqdm(self.train_dataloader):

            # Load data from trainloader

            Xb, _ = sample_batched
            Xb = Xb.squeeze().cuda()
            Xb = self.pre(Xb)
            Xb = (self.Spec(Xb)+1e-8).log()
            N_ += Xb.size(0)
            mean += Xb.mean(2).sum(0)
            variance += Xb.var(2).sum(0)
        mean = mean / N_
        variance = variance / N_
        standard_dev = torch.sqrt(variance)

        # Save global mean/var
        torch.save(mean,"/data_a11/mayi/project/CAM/IRM/exp/mean.pt")
        torch.save(standard_dev,"/data_a11/mayi/project/CAM/IRM/exp/std.pt")


    def train(self, weight, resume_epoch=None):
        ## Init training
        # Resume from last epoch
        if resume_epoch != None:
            print('start from epoch',resume_epoch)
            start_epoch = resume_epoch
        else:
            start_epoch = 1
            with open(f"{self.PROJECT_DIR}/train_loss.csv", "w+") as TRAIN_LOSS:
                TRAIN_LOSS.write(f"epoch,batch,loss\n")
            with open(f"{self.PROJECT_DIR}/valid_loss.csv", "w+") as VALID_LOSS:
                VALID_LOSS.write(f"epoch,file_name,loss,snr,seg_snr,pesq\n")

        ## Load model to device
            print('begin to train')
            if torch.cuda.device_count() >= 1:
            
                self.model = nn.DataParallel(self.model).cuda()

        ## Train and validate
        for epoch in range(start_epoch+1, self.config["num_epochs"]+1):

            self.__set_models_to_train_mode()
            self.__train_epoch(epoch, weight)
            self.__set_models_to_eval_mode()
            self.__validation_epoch(epoch, weight)
