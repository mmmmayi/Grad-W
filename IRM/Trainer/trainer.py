import torch,math
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import librosa.display
import time
import os, torchaudio
import numpy as np
from Models.models import PreEmphasis
from Models.TDNN import Speaker_resnet, ArcMarginProduct
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
        
        self.mean = torch.load("exp/resnet_mean.pt").float().cuda()
        self.std = torch.load("exp/resnet_mean.pt").float().cuda()

        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Setup logger
        comment = project_dir.split('/')[-1]
        self.writer = SummaryWriter(comment=comment)
        self.logger = get_logger(f"{self.PROJECT_DIR}/LOG")
        self.logger.info(self.config)
        self.speaker = Speaker_resnet().cuda()
        projection = ArcMarginProduct().cuda()
        self.speaker.add_module("projection", projection)
        path = 'exp/resnet.pt'
        checkpoint = torch.load(path)

        self.speaker.load_state_dict(checkpoint, strict=False)
        state_dict = self.speaker.state_dict()
        #for name,para in state_dict.items():
            #print(name)
        #for p in self.speaker.parameters():
            #p.requires_grad = False
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
        clip_length = 100*self.dur
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
        relu = nn.ReLU()
        running_mse, running_preserve, running_remove, running_enh, running_diff = 0,0,0,0,0
        i_batch,diff_batch = 0,0
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            # Load data from trainloader
            Xb, target_spk, clean, correct_spk  = sample_batched
            
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda()
            clean = clean.reshape(B,W).cuda()
            target_spk = target_spk.reshape(B)
            with torch.no_grad():
                #Xb = self.pre(Xb)
                Xb = (self.Spec(Xb)+1e-8).log()           
                #clean = self.pre(clean)
                clean = (self.Spec(clean)+1e-8).log()     
 
                frame_len = Xb.shape[-1]
                start, end = self._clip_point(frame_len) 
                Xb = Xb[:,:,start:end]
                clean = clean[:,:,start:end] 
                mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()
                mel_c = (self.Mel_scale(clean.exp())+1e-6).log()
                feature = mel_n.requires_grad_()
            mask = self.model(mel_n,mel_c)
            if correct_spk.item() is False:
                train_loss = torch.mean(torch.pow(mask-0.5,2))
                running_diff += train_loss.item()
                diff_batch += 1
            else:
                score = self.speaker(feature, target_spk.cuda(), 'score')
                self.speaker.zero_grad()
                yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
                SaM = self.vari_sigmoid(yb,50)
                '''
                for i in range(16):
                    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                    librosa.display.specshow(mel_c[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])
                    img = librosa.display.specshow(yb[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'.png')
                    plt.close()
                quit()
                '''

                inverse_mask = 1-mask
                mse_loss = self.mse(mask,SaM)
                enh_loss = torch.mean(self.vari_ReLU(0-yb,self.ratio)*torch.pow(relu(mask-SaM),2)+self.vari_ReLU(yb,self.ratio)*torch.pow(relu(SaM-mask),2))
                preserve_score = torch.mean(self.speaker(mel_c+(mask+0.5).log(), target_spk.cuda(), 'score'))
                remove_score = torch.mean(self.speaker(mel_c+(inverse_mask+0.5).log(), target_spk.cuda(), 'score'))
                train_loss = mse_loss-preserve_score+weight*enh_loss+0.01*remove_score
                running_mse += mse_loss.item()
                running_preserve += 0-preserve_score.item()
                running_remove += remove_score.item()
                running_enh += enh_loss.item()
                i_batch += 1

            self.optimizer.zero_grad()

            try:
                train_loss.backward()
            except RuntimeError as e:
                print(e)
                quit()
            
            self.optimizer.step()
            torch.cuda.empty_cache()
            self.logger.info("Loss of minibatch {}-th/{}: {}, lr:{}".format(i_batch+diff_batch, len(self.train_dataloader), train_loss.item(), self.optimizer.param_groups[0]['lr']))
        #if epoch%10 ==0:
            #self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.8
        if epoch%10 ==0:
            torch.save(self.model, f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
        ave_mse_loss = running_mse / i_batch
        ave_preserve_loss = running_preserve / i_batch
        ave_remove_loss = running_remove / i_batch
        ave_enh_loss = running_enh / i_batch
        ave_diff_loss = running_diff / diff_batch
        self.writer.add_scalar('Train/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Train/preserve', ave_preserve_loss, epoch)
        self.writer.add_scalar('Train/remove', ave_remove_loss, epoch)
        self.writer.add_scalar('Train/enh', ave_enh_loss, epoch)
        self.writer.add_scalar('Train/diff', ave_diff_loss, epoch)
        self.logger.info("Epoch:{}, loss = {}".format(epoch, ave_mse_loss+ave_preserve_loss+ave_enh_loss+ave_remove_loss)) 
        self.logger.info("*" * 50)
    

    
    def __set_models_to_eval_mode(self):
        self.model.eval()

    def __validation_epoch(self, epoch, weight):
        start_time = time.time()
        running_mse_loss, running_preserve_loss, running_remove_loss, running_enh_loss, running_diff_loss  = 0.0, 0.0, 0.0, 0.0, 0.0
        i_batch,diff_batch = 0,0
        relu = nn.ReLU()
        
        print("Validating...")
        for sample_batched in tqdm(self.validation_dataloader):
            Xb, target_spk, clean, correct_spk = sample_batched
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
                start, end = self._clip_point(frame_len)
                Xb = Xb[:,:,start:end]
                clean = clean[:,:,start:end]

                mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()
                mel_c = (self.Mel_scale(clean.exp())+1e-6).log()
 

                feature = mel_n.requires_grad_()
            mask = self.model(mel_n, mel_c)
            if correct_spk.item() is False:
                val_loss = torch.mean(torch.pow(mask-0.5,2))
                running_diff_loss += val_loss.item()
                diff_batch += 1
            else:
                score = self.speaker(feature, target_spk.cuda(), 'score')
                self.speaker.zero_grad()

                yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
                SaM = self.vari_sigmoid(yb,50)
            #frame_len = mel_c.shape[-1]
            #if frame_len%8>0:
                #pad_num = math.ceil(frame_len/8)*8-frame_len
                #pad = torch.nn.ZeroPad2d((0,pad_num,0,0))
                #mel_c_ = pad(mel_c)
            
                inverse_mask = 1-mask
            
                mse_loss = self.mse(mask, yb)
                enh_loss = torch.mean(self.vari_ReLU(0-yb,self.ratio)*torch.pow(relu(mask-SaM),2)+self.vari_ReLU(yb,self.ratio)*torch.pow(relu(SaM-mask),2))
                preserve_score = torch.mean(self.speaker(mel_c+(mask+0.5).log(), target_spk.cuda(), 'score'))
                remove_score = torch.mean(self.speaker(mel_c+(inverse_mask+0.5).log(), target_spk.cuda(), 'score'))
                running_mse_loss += mse_loss.item()
                running_preserve_loss += 0-preserve_score.item()
                running_enh_loss += enh_loss.item()
                running_remove_loss += remove_score.item()
            #running_remove_loss += remove_score.item()
                i_batch += 1
            
        ave_mse_loss = running_mse_loss / i_batch
        ave_preserve_loss = running_preserve_loss / i_batch
        ave_remove_loss = running_remove_loss / i_batch
        ave_enh_loss = running_enh_loss / i_batch
        ave_diff_loss = running_diff_loss / diff_batch
        end_time = time.time()
        self.writer.add_scalar('Validation/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Validation/preserve', ave_preserve_loss, epoch)
        self.writer.add_scalar('Validation/remove', ave_remove_loss, epoch)
        self.writer.add_scalar('Validation/enh', ave_enh_loss, epoch)
        self.writer.add_scalar('Validation/diff', ave_diff_loss, epoch)
        self.logger.info(f"Time used for this epoch validation: {end_time - start_time} seconds")
        self.logger.info("Epoch:{}, average loss = {}".format(epoch, ave_mse_loss+ave_preserve_loss+ave_remove_loss+ave_enh_loss))

    def _get_global_mean_variance(self):
        mean = 0.0
        variance = 0.0
        N_ = 0
        for sample_batched in tqdm(self.train_dataloader):

            # Load data from trainloader

            Xb, target_spk, clean, check = sample_batched
            Xb = Xb.squeeze().cuda()
            #Xb = self.pre(Xb)
            Xb = (self.Spec(Xb)+1e-8).log()
            N_ += Xb.size(0)
            mean += Xb.mean(2).sum(0)
            variance += Xb.var(2).sum(0)
        mean = mean / N_
        variance = variance / N_
        standard_dev = torch.sqrt(variance)

        # Save global mean/var
        torch.save(mean,"/data_a11/mayi/project/SIP/IRM/exp/resnet_mean.pt")
        torch.save(standard_dev,"/data_a11/mayi/project/SIP/IRM/exp/resnet_std.pt")


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

            #self.__set_models_to_train_mode()
            #self.__train_epoch(epoch, weight)
            self.__set_models_to_eval_mode()
            self.__validation_epoch(epoch, weight)
