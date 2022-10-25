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
        
        self.mean = torch.load("/data_a11/mayi/project/CAM/IRM/exp/mean.pt").float().cuda()
        self.std = torch.load("/data_a11/mayi/project/CAM/IRM/exp/std.pt").float().cuda()

        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Setup logger
        comment = project_dir.split('/')[-1]
        self.writer = SummaryWriter(comment=comment)
        self.logger = get_logger(f"{self.PROJECT_DIR}/LOG")
        self.logger.info(self.config)
        self.speaker = Speaker_TDNN().cuda()
        self_state = self.speaker.state_dict()
        path = '/data_a11/mayi/project/ECAPATDNN-analysis/exps/pretrain.model'
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
            #target_spk = target_spk.squeeze().cuda()
            with torch.no_grad():
                Xb = self.pre(Xb)
                Xb = (self.Spec(Xb)+1e-8).log()           
                clean = self.pre(clean)
                clean = (self.Spec(clean)+1e-8).log()     
                #Xb = (self.MelSpec(Xb)+1e-6).log()
 
                frame_len = Xb.shape[-1]
                start, end = self._clip_point(frame_len) 
            #yb = yb[:,:,start:end]
                Xb = Xb[:,:,start:end]
                clean = clean[:,:,start:end]
                mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()
                mel_c = (self.Mel_scale(clean.exp())+1e-6).log()
                feature = mel_c.requires_grad_()

                Xb_input = (Xb-self.mean)/self.std
 
            #print('label device:',target_spk.device)
            score = self.speaker(feature, target_spk.cuda(), 'score')
            self.speaker.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            
            self.optimizer.zero_grad()
            
            cam = self.model(mel_n, Xb_input, target_spk)
            pred_spec = (cam+1e-8).log()+Xb
            pred_mel = (self.Mel_scale(pred_spec.exp())+1e-6).log()
            mse_loss = self.mse(pred_spec, clean)/B
            #print(torch.amax(self.vari_ReLU(0-yb,self.ratio),dim=(1,2)).unsqueeze(-1).unsqueeze(-1))
            #quit()
            disto_loss = self.vari_ReLU(0-yb,self.ratio)*torch.pow(relu(pred_mel-mel_c),2)+self.vari_ReLU(yb,self.ratio)*torch.pow(relu(mel_c-pred_mel),2)
            disto_loss = torch.mean(disto_loss)

            train_loss = mse_loss+disto_loss*weight
            '''
            neg = self.vari_ReLU(0-yb,self.ratio)
            pos = self.vari_ReLU(yb,self.ratio)
            for i in range(16):
                fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
                librosa.display.specshow(mel_c[i].squeeze().detach().cpu().numpy(), x_axis=None, ax=ax[0])
                img2 = librosa.display.specshow(neg[i].detach().cpu().numpy(),x_axis=None, ax=ax[1] )
                librosa.display.specshow(pos[i].squeeze().detach().cpu().numpy(),x_axis=None, ax=ax[2])
                plt.savefig('/data_a11/mayi/project/CAM/IRM/exp/debug/'+str(i)+'_'+str(self.ratio)+'.png')
                plt.close()
            quit()
            '''
            running_mse += mse_loss.item()
            running_cos += disto_loss.item()
            # Backward
            #print(self.model.module.fconn.weight.grad)
            
            try:
                train_loss.backward()
            except RuntimeError as e:
                print(e)
                #torch.save(cam,'exp/debug/cam.pt')
                #torch.save(self.model, 'exp/debug/model.pt')
                quit()
            
            #print(self.model.module.fconn.weight.grad)
            #for name, param in self.model.named_parameters():
                #if param.requires_grad:
                    #print(name,param.grad)
            #print('before',self.model.module.fconn.weight)
            self.optimizer.step()
            torch.cuda.empty_cache()
            #print('grad',self.model.module.fconn.weight.grad)
            #print('after',self.model.module.fconn.weight)
            # Log
            i_batch += 1
            #self.writer.add_graph(self.model, (Xb, Xb_input, target_spk))
            self.logger.info("Loss of minibatch {}-th/{}: {}, lr:{}".format(i_batch, len(self.train_dataloader), train_loss.item(), self.optimizer.param_groups[0]['lr']))
            #with open(f"{self.PROJECT_DIR}/train_loss.csv", "a+") as LOSS:
                #LOSS.write(f"{epoch},{i_batch},{train_loss.item()}\n")
        # Save model after each epoch
        if epoch%10 ==0:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.8
        if epoch%1 ==0:
            torch.save(self.model, f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
        # Log after each epoch
        ave_mse_loss = running_mse / i_batch
        ave_cos_loss = running_cos / i_batch
        '''
        if ave_epoch_loss>self.last_loss:
            self.time +=1
        else:
            self.time = 0
        if self.time == 2:
            self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/2
            self.time = 0
        self.last_loss = ave_epoch_loss
        '''
        end_time = time.time()
        self.writer.add_scalar('Train/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Train/cos', ave_cos_loss, epoch)
        self.logger.info("Epoch:{}, loss = {}".format(epoch, ave_mse_loss+ave_cos_loss)) 
        self.logger.info(f"Time used for this epoch training: {end_time - start_time} seconds")
        self.logger.info("*" * 50)
    
    def logistic(self, score, score_nor, target, target_nor, base, weight):
        m = nn.Sigmoid()
        d_score_pos = m(score-base)
        d_score_neg = m(base-score_nor)
        d_target_pos = m(target-base)
        d_target_neg = m(base-target_nor)
        loss = -d_target_pos*(d_score_pos.log())-d_target_neg*(d_score_neg.log())
        return torch.mean(loss)

    def CE(self, pred, label):
        pred = torch.permute(pred, (0,2,1))
        label = torch.permute(label, (0,2,1))
        weight = torch.permute(weight, (0,2,1))
        B,T,F = pred.shape
        pred = pred.unsqueeze(1).expand(B,T,T,F)
        label = label.unsqueeze(2).expand(B,T,T,F)
 
        diff = 0-torch.abs(torch.sum(pred-label, -1))
        aux_label = torch.arange(0,T).unsqueeze(0).expand(B,T).cuda()
        result = self.ce(diff,aux_label)
        #print(result)
        return result
    def l2norm(self,cam,yb):
        cam_l2 = torch.norm(cam,p=2,dim=1)
        yb_l2 = torch.norm(yb,p=2,dim=1)
        diff_l2 = cam_l2-yb_l2
        norm = torch.where(diff_l2>0, cam_l2, torch.tensor(0, dtype=diff_l2.dtype).cuda())
        return norm.mean()
    
    def highlight_mse(self,cam,yb,Xb):
        
        
        B,H,W = yb.shape
        yb_mean = torch.mean(yb, dim=(1,2)).reshape(B,1,1).repeat(1,H,W).cuda()
        encrease = torch.where(yb>(yb_mean+0.01), 1, 0)
        encrease = (encrease+1e-6).log()+Xb
        suppress = torch.where(yb<(yb_mean-0.01),1,0)
        suppress = (suppress+1e-6).log()+Xb
        '''
        for i in range(10):
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            librosa.display.specshow(suppress[i].detach().cpu().numpy(),x_axis=None, ax=ax[0])
            librosa.display.specshow(encrease[i].detach().cpu().numpy(),x_axis=None, ax=ax[1])
            librosa.display.specshow(Xb[i].detach().cpu().numpy(),x_axis=None, ax=ax[2])

        #fig.colorbar(img2, ax=ax)
            plt.savefig('/data_a11/mayi/project/CAM/IRM/exp/debug/'+str(i)+'.png')
            plt.close()
 
        quit()
        '''
        pos = self.mse((encrease+(cam+1e-6).log()).exp(),encrease.exp())
        neg = torch.mean((suppress+(cam+1e-6).log()).exp())
        return pos, neg

    def compute_highlight(self,mask):
        batch, frequency, frame = mask.shape
        loss_mask = torch.zeros((batch, frequency, frame)).cuda()
        min_ = torch.min(mask)
        max_ = torch.max(mask)
        up = torch.mean(mask, (1,2))+min(0.004,max_-0.001)
        below =torch.mean(mask, (1,2))-max(0.002,min_+0.001)
        up = up.reshape(batch,-1).unsqueeze(-1).expand(batch, frequency, frame)
        below = below.reshape(batch,-1).unsqueeze(-1).expand(batch, frequency, frame)
        index_up = mask[:].ge(up[:])
        index_below = mask[:].le(below[:])
        num=torch.sum(index_up == True)+torch.sum(index_below == True)
        loss_mask[index_up]=1
        loss_mask[index_below]=1
        return loss_mask,num


    def __set_models_to_eval_mode(self):
        self.model.eval()

    def computing_ranking(self, cam, yb):
        loss = 0
        for i in range(1):
            pred = (cam[:,:,i]+cam[:,:,i+1]).squeeze()
            gt = (yb[:,:,i]+yb[:,:,i+1]).squeeze()
            gt_scores, gt_ixs = torch.sort(gt, 1, descending=True)
            gt_scores = gt_scores.squeeze()
            gt_diff = gt_scores.unsqueeze(2)-gt_scores.unsqueeze(1)

            pred_scores = pred.gather(1, gt_ixs.squeeze())

            pred_diff = pred_scores.unsqueeze(2)-pred_scores.unsqueeze(1)
            gt_mask = torch.where(gt_diff < 0, torch.zeros_like(gt_diff), torch.ones_like(gt_diff))
            pred_scores_mask = pred_diff * gt_mask

            pred_mask = torch.where(pred_diff < 0, -1 * torch.ones_like(pred_diff), torch.zeros_like(pred_diff))
            np.set_printoptions(threshold=np.inf)
            pred_scores_mask = pred_scores_mask * pred_mask
            loss += pred_scores_mask.sum(dim=1).mean()
        return loss
 
    #@torch.no_grad()
    def __validation_epoch(self, epoch, weight):
        '''The batch size of validation dataloader must be 1'''
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
                #start, end = self._clip_point(frame_len)
                #Xb = Xb[:,:,start:end]
                #clean = clean[:,:,start:end]
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
            #if torch.isnan(val_loss):
               # print(check)
            #val_mse = self.mse(cam_contribution, yb_contribution)
           
            #for i in range(1,len(repre)):
                #val_loss += self.mse(repre[i],target_repre[i])
                #print(val_loss)
                #val_loss += self.mse(repre_nor[i], target_repre_nor[i])
                #print(val_loss)
            #val_loss += self.mse(prd_spec,yb_spec)
            #val_loss += self.mse(prd_spec_nor, yb_spec_nor)
            #quit()
            #reg = self.mse(cam.squeeze(), yb_mean)
            #print('x_loss'+str(torch.mean(x_loss).item())+'reg:'+str(reg.item()))

            #print(torch.mean(x_loss))
            running_mse_loss += mse_loss.item()
            running_cos_loss += disto_loss.item()
            #highlight_loss = self.highlight_mse(cam,yb)
            #norm_loss = self.l2norm(cam,yb)
            
            #val_loss = 100*(weight*highlight_loss+val_loss/(num*frames))


            i_batch += 1
            
            '''
            print(val_loss)
            show_output = cam[0].detach()
            show_label = yb[0].detach()
            show_X = Xb[0].detach()
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

            librosa.display.specshow(show_output.cpu().numpy(),x_axis=None, ax=ax[0])
            librosa.display.specshow(show_label.cpu().numpy(), x_axis=None, ax=ax[1])
            librosa.display.specshow(show_X.cpu().numpy(),x_axis=None, ax=ax[2])
            plt.savefig('debug.png')
            quit()
            '''                                                                                                          
            # Log after each epoch
        ave_cos_loss = running_cos_loss / i_batch
        ave_mse_loss = running_mse_loss / i_batch
        #print(ave_cos_loss)
        #print(ave_mse_loss)
        end_time = time.time()
        self.writer.add_scalar('Validation/mse', ave_mse_loss, epoch)
        self.writer.add_scalar('Validation/cos', ave_cos_loss, epoch)
        self.logger.info(f"Time used for this epoch validation: {end_time - start_time} seconds")
        self.logger.info("Epoch:{}, average loss = {}".format(epoch, ave_cos_loss+ave_mse_loss))
        #self.logger.info("*" * 50)

    def _get_global_mean_variance(self,mag_type):
        mean = 0.0
        variance = 0.0
        N_ = 0
        #print('test for dataloader:',len(self.train_dataloader))
        for sample_batched in tqdm(self.train_dataloader):

            # Load data from trainloader

            Xb, _ = sample_batched
            Xb = Xb.squeeze().cuda()
            Xb = self.pre(Xb)
            Xb = (self.Spec(Xb)+1e-8).log()
            N_ += Xb.size(0)
            mean += Xb.mean(2).sum(0)
            variance += Xb.var(2).sum(0)
#            print('Xb shape:{},N_:{},mean shape:{}'.format(Xb.shape,N_,mean.shape))
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
            #self.model.load_state_dict(loaded_states.module['model_state_dict'])
        # Train from epoch 1
        else:
            start_epoch = 1
            # Set up csv
            with open(f"{self.PROJECT_DIR}/train_loss.csv", "w+") as TRAIN_LOSS:
                TRAIN_LOSS.write(f"epoch,batch,loss\n")
            with open(f"{self.PROJECT_DIR}/valid_loss.csv", "w+") as VALID_LOSS:
                VALID_LOSS.write(f"epoch,file_name,loss,snr,seg_snr,pesq\n")

        ## Load model to device
            print('begin to train')
            if torch.cuda.device_count() >= 1:
            
                self.model = nn.DataParallel(self.model).cuda()

        ## Train and validate
        #self.__set_models_to_eval_mode()
        #self.__validation_epoch(1, weight)
        #quit()
        for epoch in range(start_epoch+1, self.config["num_epochs"]+1):
        #for epoch in range(2,101,10):

            self.__set_models_to_train_mode()
            self.__train_epoch(epoch, weight)
            #if epoch % 20 == 0:
            #self.model = torch.load(f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
            self.__set_models_to_eval_mode()
            self.__validation_epoch(epoch, weight)
