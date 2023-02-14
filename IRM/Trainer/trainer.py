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
import torch.nn.functional as nnf
import torch.distributed as dist
class IRMTrainer():
    def __init__(self,
            config, project_dir,
            model, optimizer, loss_fn, dur,
            train_dl, validation_dl, mode, ratio, 
            local_rank,w_p,w_n,w_hn):
        # Init device
        self.config = config
        self.w_p = w_p
        self.w_n = w_n
        self.w_hn = w_hn
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
        self.auxl = self.auxl.cuda()
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
        reg = torch.where(x > max*th, x, torch.tensor(0, dtype=x.dtype).cuda())
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

    def cw_loss(self,logits, target_spk, device, targeted=True, t_conf=15, nt_conf=0.5):
        depth = logits.shape[-1]
        one_hot_labels = torch.zeros(target_spk.size(0), depth).cuda()
        
        one_hot_labels =torch.zeros(target_spk.size(0), depth).cuda().scatter_(1, target_spk.view(-1, 1).data, 1)
        this = torch.sum(logits*one_hot_labels, 1)
        other_best, _ = torch.max(logits*(1.-one_hot_labels) - 12111*one_hot_labels, 1)   # subtracting 12111 from selected labels to make sure that they dont end up a maximum
        #print('best',this)
        #print('second',other_best)
        #print('==========')
        t = nnf.relu(other_best - this + t_conf)
        nt = nnf.relu(this - other_best + nt_conf)
        if isinstance(targeted, (bool, int)):
            return torch.mean(t) if targeted else torch.mean(nt)

    def accuracy(self, output, target):
        maxk = 1
        batch_size, T,F = target.shape
        _, pred = output.topk(maxk, 1, True, True)
        pred = torch.sum(pred,1)
        correct = pred.eq(target)
        correct = correct.float().sum()
        res = correct/(batch_size*T*F)
        return res

    def ce(self, output, target, yb):
        output_ = 1-output
        output = torch.cat((output_.unsqueeze(1), output.unsqueeze(1)),1)
        B,C,T,F = output.size()
        #weight_ = torch.where(target==0,torch.tensor(0.05, dtype=output.dtype).cuda().to(device),torch.tensor(0.95, dtype=output.dtype).cuda().to(device))
        '''
        box = torch.ones((3, 3), requires_grad=False)
        box = box[None, None, ...].repeat(1, 1, 1, 1).cuda()
        weight = nnf.conv2d(target.unsqueeze(1).float(), box, padding=1,groups=1)
        for i in range(9):
            weight = torch.where(weight==i,0.05+i*0.3,weight)
        
        weight_ = torch.where(target.unsqueeze(1)==1,4,weight)
        '''
        min = torch.amin(yb,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
        weight = torch.where(yb<self.config["th"]*min,torch.tensor(self.w_hn, dtype=yb.dtype).cuda(),torch.tensor(self.w_n, dtype=yb.dtype).cuda())
        weight_ = torch.where(target==1,self.w_p,weight)
        output = output.reshape(B,C,T*F)
        target = target.reshape(B,1,T*F)
        weight = weight_.reshape(B,T*F).detach()
        output_class = torch.gather(output,1,target).squeeze()
        ce = -torch.log(output_class)*weight
        ce = ce.reshape(-1)
        weight = weight.reshape(-1)
        idx_p = (weight==self.w_p).nonzero().squeeze()
        idx_hn = (weight==self.w_hn).nonzero().squeeze()
        idx_n = (weight==self.w_n).nonzero().squeeze()
        ce_p = torch.gather(ce,0,idx_p)
        ce_n = torch.gather(ce,0,idx_n)
        ce_hn = torch.gather(ce,0,idx_hn)
        return torch.sum(ce_p)/torch.sum(weight), torch.sum(ce_n)/torch.sum(weight), torch.sum(ce_hn)/torch.sum(weight), weight_

    def recall(self, output, weight):
        output_ = 1-output
        output = torch.cat((output_.unsqueeze(1), output.unsqueeze(1)),1)       
        maxk = 1
        batch_size, T,F = weight.shape
        _, pred = output.topk(maxk, 1, True, True)
        pred = torch.sum(pred,1)
        pred = pred.reshape(-1)
        p_idx = torch.nonzero(pred).squeeze()
        t_idx = (weight.reshape(-1) == self.w_p).nonzero().squeeze()
        tpr = len(np.intersect1d(p_idx.detach().cpu(),t_idx.detach().cpu()))/t_idx.numel()
        n_idx = (pred == 0).nonzero().squeeze()
        f_idx = (weight.reshape(-1) == self.w_hn).nonzero().squeeze()
        tnr = len(np.intersect1d(n_idx.detach().cpu(),f_idx.detach().cpu()))/f_idx.numel()
        return tpr, tnr

    def train_epoch(self, epoch, weight, device, loader_size, scheduler):
        relu = nn.ReLU()
        running_p, running_b, running_n, running_tpr, running_tnr, running_mse, running_tht = 0,0,0,0,0,0,0
        i_batch,diff_batch = 0,0
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            cur_iter = (epoch - 1) * loader_size + i_batch
            scheduler.step(cur_iter)
            # Load data from trainloader
            Xb, target_spk, clean, correct_spk  = sample_batched
            #print(Xb.device)
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda()
            clean = clean.reshape(B,W).cuda()
            target_spk = target_spk.reshape(B).cuda()
            logits, feature = self.model(clean)
            '''
            if device==0:
                for i in range(10):
                    encoder_0 = torch.mean(encoder[0][i,:,:,:],0).squeeze()
                    encoder_1 = torch.mean(encoder[1][i,:,:,:],0).squeeze()
                    encoder_2 = torch.mean(encoder[2][i,:,:,:],0).squeeze()
                    encoder_3 = torch.mean(encoder[3][i,:,:,:],0).squeeze()
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    #temp2 = self.vari_ReLU(yb,self.ratio,device)
                    img = librosa.display.specshow(encoder_0.detach().cpu().squeeze().numpy(),x_axis=None)
                    fig.colorbar(img, ax=ax)
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'0.png')
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    img = librosa.display.specshow(encoder_1.detach().cpu().squeeze().numpy(),x_axis=None)
                    fig.colorbar(img, ax=ax)
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'1.png')
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    img = librosa.display.specshow(encoder_2.detach().cpu().squeeze().numpy(),x_axis=None)
                    fig.colorbar(img, ax=ax)
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'2.png')
                    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
                    img = librosa.display.specshow(encoder_3.detach().cpu().squeeze().numpy(),x_axis=None)
                    fig.colorbar(img, ax=ax)
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'3.png')
 

            quit()
            '''

            feature = feature.detach().requires_grad_()
            score = self.auxl(feature, target_spk, 'score')
            self.auxl.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            max = torch.amax(yb,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
            SaM = torch.where(yb>self.config["th"]*max,torch.tensor(1, dtype=yb.dtype).cuda(),torch.tensor(0, dtype=yb.dtype).cuda())
            SaM =SaM.long()
            #SaM = self.vari_sigmoid(yb,50)
            ce_p, ce_b, ce_n, weight = self.ce(logits,SaM,yb)
            #print(self.bce(logits,SaM))
            '''
            if device==0:
                for i in range(10):
                    mask_ = np.sort(yb[i,:,:].detach().cpu().squeeze().numpy(),axis=None).squeeze()
                    x = np.arange(len(mask_))
                    plt.plot(x, mask_, color='blue', label='predict')
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'sort.png')

                    plt.close()
 
                    temp = 0-SaM
                    #temp2 = self.vari_ReLU(yb,self.ratio,device) 
                    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
                    librosa.display.specshow(feature[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])
                    librosa.display.specshow(SaM[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
#                    img = librosa.display.specshow(mask[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
                    img = librosa.display.specshow(weight[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
                    #print(weight[i,:,:])
                    fig.colorbar(img, ax=ax)

                    #img = librosa.display.specshow(yb[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'varied.png')
                    plt.close()
           
            quit() 
            '''
            #inverse_mask = 1-mask
            
            tpr, tnr = self.recall(logits,weight)    
            #mse = self.mse(mask, SaM.float())
            train_loss = ce_p+ce_b+ce_n
            if torch.isnan(train_loss):
                torch.save(feature,'/data_a11/mayi/project/SIP/IRM/exp/debug/feature.pt')
                state_dict = self.model.state_dict()
                torch.save(state_dict,'/data_a11/mayi/project/SIP/IRM/exp/debug/model.pt')
                quit()
            running_p += ce_p.item()
            running_b += ce_b.item()
            running_n += ce_n.item()
            #running_mse += mse.item()
            #running_preserve += preserve_score.item()
            #running_remove += remove_score.item()
            running_tpr += tpr
            running_tnr += tnr
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
            ave_p = running_p / i_batch
            ave_b = running_b / i_batch
            ave_n = running_n / i_batch
            #ave_mse = running_mse / i_batch
            ave_tnr_loss = running_tnr / i_batch
            ave_tpr_loss = running_tpr / i_batch
            self.writer.add_scalar('Train/p', ave_p, epoch)
            self.writer.add_scalar('Train/b', ave_b, epoch)
            self.writer.add_scalar('Train/n', ave_n, epoch)
            #self.writer.add_scalar('Train/mse', ave_mse, epoch)
            self.writer.add_scalar('Train/tnr', ave_tnr_loss, epoch)
            self.writer.add_scalar('Train/tpr', ave_tpr_loss, epoch)
            self.logger.info("Epoch:{}".format(epoch)) 
            self.logger.info("*" * 50)
    

    
    def set_models_to_eval_mode(self):
        self.model.eval()

    def validation_epoch(self, epoch, weight, device):
        start_time = time.time()
        running_p, running_b, running_n, running_mse_loss, running_diff_loss, running_tnr_loss, running_tpr_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        i_batch,diff_batch = 0,0
        relu = nn.ReLU()
        
        
        for sample_batched in tqdm(self.validation_dataloader):
            Xb, target_spk, clean, correct_spk = sample_batched
            _, B, _, W = Xb.shape
            Xb = Xb.reshape(B,W).cuda()
            clean = clean.reshape(B,W).cuda()
            target_spk = target_spk.squeeze().cuda()
           
            logits, feature = self.model(clean)
            feature = feature.requires_grad_()
            score = self.auxl(feature, target_spk, 'score')
            self.auxl.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            max = torch.amax(yb,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
            SaM = torch.where(yb>self.config["th"]*max,torch.tensor(1, dtype=yb.dtype).cuda(),torch.tensor(0, dtype=yb.dtype).cuda())
            SaM =SaM.long()

                #SaM_frame = self.vari_sigmoid(yb_frame,50)
            #frame_len = mel_c.shape[-1]
            #if frame_len%8>0:
                #pad_num = math.ceil(frame_len/8)*8-frame_len
                #pad = torch.nn.ZeroPad2d((0,pad_num,0,0))
                #mel_c_ = pad(mel_c)
            
            #inverse_mask = 1-mask
            ce_p, ce_b, ce_n,weight = self.ce(logits,SaM,yb)
            #mse = self.mse(mask, SaM.float())
            tpr,tnr = self.recall(logits,weight)
            #enh_loss = torch.mean(self.vari_ReLU(0-yb,self.ratio,device)*torch.pow(mask-SaM,2)+self.vari_ReLU(yb,self.ratio,device)*torch.pow(SaM-mask,2))
            #logits = self.auxl(feature, target_spk, 'loss', mask)
            #preserve_score = self.cw_loss(logits, target_spk, device, True)
            #logits = self.auxl(feature, target_spk, 'loss', inverse_mask)
            #remove_score = self.cw_loss(logits,target_spk,device, False)
            running_p += ce_p.item()
            #running_mse_loss += mse.item()
            running_b += ce_b.item()
            running_n += ce_n.item()
            running_tnr_loss += tnr
            running_tpr_loss += tpr
            #running_remove_loss += remove_score.item()
            #running_remove_loss += remove_score.item()
            i_batch += 1
        if device==0:    
            ave_p = running_p / i_batch
            ave_b = running_b / i_batch
            ave_n = running_n / i_batch
            #ave_mse = running_mse_loss / i_batch
            ave_tnr_loss = running_tnr_loss / i_batch
            ave_tpr_loss = running_tpr_loss / i_batch
            end_time = time.time()
            self.writer.add_scalar('Validation/p', ave_p, epoch)
            self.writer.add_scalar('Validation/b', ave_b, epoch)
            #self.writer.add_scalar('Validation/mse', ave_mse, epoch)
            self.writer.add_scalar('Validation/n', ave_n, epoch)
            self.writer.add_scalar('Validation/tpr', ave_tpr_loss, epoch)
            self.writer.add_scalar('Validation/tnr', ave_tnr_loss, epoch)
            print('tpr:{}, tnr:{}'.format(ave_tpr_loss,ave_tnr_loss))
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
