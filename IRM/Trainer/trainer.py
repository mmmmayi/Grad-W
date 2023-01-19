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
import torch.distributed as dist
class IRMTrainer():
    def __init__(self,
            config, project_dir,
            model, optimizer, loss_fn, dur,
            train_dl, validation_dl, mode, ratio, local_rank):
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
        # Init project dir
        self.PROJECT_DIR = project_dir
        if local_rank==0:
            if not os.path.exists(f"{self.PROJECT_DIR}/models"):
                os.makedirs(f"{self.PROJECT_DIR}/models")
        
        self.mean = torch.load("exp/resnet_mean.pt").float().cuda()
        self.std = torch.load("exp/resnet_mean.pt").float().cuda()

        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Setup logger
        comment = project_dir.split('/')[-1]
        if local_rank==0:
            self.writer = SummaryWriter(comment=comment)
            self.logger = get_logger(f"{self.PROJECT_DIR}/LOG")
            self.logger.info(self.config)
        #for name,para in state_dict.items():
            #print(name)
        #for p in self.speaker.parameters():
            #p.requires_grad = False

            print(f"log file at: {self.PROJECT_DIR}/LOG.log")
        self.auxl = Speaker_resnet()
        projection = ArcMarginProduct()
        self.auxl.add_module("projection", projection)
        self.auxl = self.auxl.cuda().to(local_rank)
        checkpoint = torch.load('exp/resnet_5994.pt')
        self.auxl.load_state_dict(checkpoint, strict=False)
        self.auxl.eval()

    def set_models_to_train_mode(self):
        self.model.train()

    def vari_sigmoid(self,x,a):
        return 1/(1+(-a*x).exp())

    def vari_ReLU(self,x,th,device):
        assert th>0 and th<1
        relu = nn.ReLU()
        x = relu(x)
        max = torch.amax(x,dim=(1,2)).unsqueeze(-1).unsqueeze(-1)
        reg = torch.where(x > max*th, x, torch.tensor(0, dtype=x.dtype).cuda().to(device))
        return reg/max
    def get_contributing_params(self, y, top_level=True):
        nf = y.grad_fn.next_functions if top_level else y.next_functions
        for f, _ in nf:
            try:
                yield f.variable
            except AttributeError:
                print(f)
                pass  # node has no tensor
            if f is not None:
                yield from self.get_contributing_params(f, top_level=False)

    def cw_loss(self,logits, target_spk, device, targeted=True, t_conf=15, nt_conf=2):
        depth = logits.shape[-1]
        one_hot_labels = torch.zeros(target_spk.size(0), depth).cuda().to(device)
        
        one_hot_labels =torch.zeros(target_spk.size(0), depth).cuda().to(device).scatter_(1, target_spk.view(-1, 1).data, 1)
        this = torch.sum(logits*one_hot_labels, 1)
        other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        #print('best',this)
        #print('second',other_best)
        #print('==========')
        t = F.relu(other_best - this + t_conf)
        nt = F.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)

    def train_epoch(self, epoch, weight, device, loader_size, scheduler):
        relu = nn.ReLU()
        running_mse, running_preserve, running_remove, running_enh, running_diff, running_thf, running_tht = 0,0,0,0,0,0,0
        i_batch,diff_batch = 0,0
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            cur_iter = (epoch - 1) * loader_size + i_batch
            scheduler.step(cur_iter)
            # Load data from trainloader
            Xb, target_spk, clean, correct_spk  = sample_batched
            #print(Xb.device)
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda().to(device)
            clean = clean.reshape(B,W).cuda().to(device)
            target_spk = target_spk.reshape(B).to(device)
            mask,feature = self.model(clean)
            feature = feature.detach().requires_grad_()
            score = self.auxl(feature, target_spk, 'score')
            self.auxl.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            SaM = self.vari_sigmoid(yb,50)
            '''
            if device==0:
                for i in range(10):
                    temp = self.vari_ReLU(0-yb,self.ratio,device)
                    temp2 = self.vari_ReLU(yb,self.ratio,device) 
                    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
                    librosa.display.specshow(feature[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])
                    librosa.display.specshow(yb[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
                    img = librosa.display.specshow(temp[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
                    librosa.display.specshow(temp2[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[3])
                    fig.colorbar(img, ax=ax)

                    #img = librosa.display.specshow(yb[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'.png')
                    plt.close()
            quit()
            '''

            inverse_mask = 1-mask
            mse_loss = self.mse(mask,SaM)
                
            #enh_loss = torch.mean(self.vari_ReLU(0-yb,self.ratio,device)*torch.pow(mask-SaM,2)+self.vari_ReLU(yb,self.ratio,device)*torch.pow(SaM-mask,2))
            logits = self.auxl(feature, target_spk, 'loss', mask)
            preserve_score =  self.cw_loss(logits, target_spk, device,True)
            #continue
            logits = self.auxl(feature, target_spk, 'loss', inverse_mask)
            remove_score = self.cw_loss(logits,target_spk,device,False)
            train_loss = mse_loss+0.01*preserve_score+0.05*remove_score
            running_mse += mse_loss.item()
            running_preserve += preserve_score.item()
            running_remove += remove_score.item()
            #running_enh += enh_loss.item()
            i_batch += 1

            self.optimizer.zero_grad()

            try:
                train_loss.backward()
            except RuntimeError as e:
                print(e)
                quit()
            
            self.optimizer.step()
            torch.cuda.empty_cache()
            if device == 0:
                self.logger.info("Loss of minibatch {}-th/{}: {}, lr:{}".format(i_batch+diff_batch, len(self.train_dataloader), train_loss.item(), self.optimizer.param_groups[0]['lr']))
        #if epoch%10 ==0:
            #self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.8
        #if device==0:
            #if epoch%10 ==0:
                #torch.save(self.model, f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
        if device ==0:
            ave_mse_loss = running_mse / i_batch
            ave_preserve_loss = running_preserve / i_batch
            ave_remove_loss = running_remove / i_batch
            #ave_enh_loss = running_enh / i_batch
            self.writer.add_scalar('Train/mse', ave_mse_loss, epoch)
            self.writer.add_scalar('Train/preserve', ave_preserve_loss, epoch)
            self.writer.add_scalar('Train/remove', ave_remove_loss, epoch)
            #self.writer.add_scalar('Train/enh', ave_enh_loss, epoch)
            self.logger.info("Epoch:{}".format(epoch)) 
            self.logger.info("*" * 50)
    

    
    def set_models_to_eval_mode(self):
        self.model.eval()

    def validation_epoch(self, epoch, weight, device):
        start_time = time.time()
        running_mse_loss, running_preserve_loss, running_remove_loss, running_enh_loss, running_diff_loss, running_tht_loss, running_thf_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        i_batch,diff_batch = 0,0
        relu = nn.ReLU()
        
        
        for sample_batched in tqdm(self.validation_dataloader):
            Xb, target_spk, clean, correct_spk = sample_batched
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda().to(device)
            clean = clean.reshape(B,W).cuda().to(device)
            target_spk = target_spk.squeeze().to(device)
           
            mask, feature = self.model(clean)
            feature = feature.requires_grad_()
            score = self.auxl(feature, target_spk, 'score')
            self.auxl.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            SaM = self.vari_sigmoid(yb,50)
                #SaM_frame = self.vari_sigmoid(yb_frame,50)
            #frame_len = mel_c.shape[-1]
            #if frame_len%8>0:
                #pad_num = math.ceil(frame_len/8)*8-frame_len
                #pad = torch.nn.ZeroPad2d((0,pad_num,0,0))
                #mel_c_ = pad(mel_c)
            
            inverse_mask = 1-mask
            mse_loss = self.mse(mask, SaM)
            #enh_loss = torch.mean(self.vari_ReLU(0-yb,self.ratio,device)*torch.pow(mask-SaM,2)+self.vari_ReLU(yb,self.ratio,device)*torch.pow(SaM-mask,2))
            logits = self.auxl(feature, target_spk, 'loss', mask)
            preserve_score = self.cw_loss(logits, target_spk, device, True)
            logits = self.auxl(feature, target_spk, 'loss', inverse_mask)
            remove_score = self.cw_loss(logits,target_spk,device, False)
            running_mse_loss += mse_loss.item()
            running_preserve_loss += preserve_score.item()
            #running_enh_loss += enh_loss.item()
            running_remove_loss += remove_score.item()
            #running_remove_loss += remove_score.item()
            i_batch += 1
        if device==0:    
            ave_mse_loss = running_mse_loss / i_batch
            ave_preserve_loss = running_preserve_loss / i_batch
            ave_remove_loss = running_remove_loss / i_batch
            #ave_enh_loss = running_enh_loss / i_batch
            end_time = time.time()
            self.writer.add_scalar('Validation/mse', ave_mse_loss, epoch)
            self.writer.add_scalar('Validation/preserve', ave_preserve_loss, epoch)
            self.writer.add_scalar('Validation/remove', ave_remove_loss, epoch)
            #self.writer.add_scalar('Validation/enh', ave_enh_loss, epoch)
            self.logger.info(f"Time used for this epoch validation: {end_time - start_time} seconds")
            self.logger.info("Epoch:{}".format(epoch))

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


    def train(self, weight, local_rank, device=torch.device('cuda'), resume_epoch=None):
        ## Init training
        # Resume from last epoch
        if resume_epoch != None:
            if local_rank == 0:

                print('start from epoch',resume_epoch)
            start_epoch = resume_epoch
        else:
            start_epoch = 1
            with open(f"{self.PROJECT_DIR}/train_loss.csv", "w+") as TRAIN_LOSS:
                TRAIN_LOSS.write(f"epoch,batch,loss\n")
            with open(f"{self.PROJECT_DIR}/valid_loss.csv", "w+") as VALID_LOSS:
                VALID_LOSS.write(f"epoch,file_name,loss,snr,seg_snr,pesq\n")

        ## Load model to device
            if local_rank == 0:

                print('begin to train')
    #        if torch.cuda.device_count() >= 1:
            
    #            self.model = nn.DataParallel(self.model).cuda()

        ## Train and validate
        for epoch in range(start_epoch+1, self.config["num_epochs"]+1):
            dist.barrier()
            self.__set_models_to_train_mode()
            self.train_epoch(epoch, weight, local_rank)
            if local_rank == 0:
                self.__set_models_to_eval_mode()
                self.__validation_epoch(epoch, weight, local_rank)
