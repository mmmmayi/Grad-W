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
            train_dl, validation_dl,  ratio, 
            local_rank,w_p,w_n,center_num):
        # Init device
        self.config = config
        self.weight = w_p
        self.w_n = w_n
        self.center_num = int(center_num)
        # Dataloaders
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl  ##to check

        # Model Params
        self.optimizer = optimizer
        self.ratio = ratio
        self.cos = loss_fn[0]
        self.mse = loss_fn[1]
        self.cos_emb = nn.CosineEmbeddingLoss()
        self.last_loss = float('inf')
        self.time = 0
        self.model = model
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
        ce = -torch.log(output_class+1e-6)
        ce = ce.reshape(-1)
        weight = weight.reshape(-1)
        idx_p = (weight==self.w_p).nonzero().squeeze()
        idx_hn = (weight==self.w_hn).nonzero().squeeze()
        idx_n = (weight==self.w_n).nonzero().squeeze()
        ce_p = torch.gather(ce,0,idx_p)
        ce_n = torch.gather(ce,0,idx_n)
        ce_hn = torch.gather(ce,0,idx_hn)
        return torch.sum(ce_p)/ce_p.numel(), torch.sum(ce_n)/ce_n.numel(), torch.sum(ce_hn)/ce_hn.numel(), weight_

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

    def mse_dis(self, test, target,dim=1):
          
        return torch.sqrt(torch.sum(torch.pow(test-target,2),dim))

    def mse_relu(self,pre,refer):
        relu=nn.ReLU()
        return torch.sum(torch.pow(relu(refer-pre),2))

    def compute_p(self, centers, test_emb, target):
        centers=centers.unsqueeze(1)
        test_emb = test_emb.unsqueeze(0)
        test_emb = test_emb.repeat(centers.shape[0],1,1)
        #print('centers',centers.shape)
        #print('test_emb',test_emb.shape)
        centers = centers.repeat(1,test_emb.shape[1],1)
        #print('centers',centers.shape)
        dis = self.mse_dis(test_emb,centers,dim=-1)
        #print('dis',dis.shape)
        dis = torch.exp(0-dis)
        dis_class = torch.gather(dis,0,target)
        #print('dis_class',dis_class)
        #print('dis_class',dis_class.shape)
        dis_sum = torch.sum(dis,dim=0).unsqueeze(0)
        #print('dis_sum',dis_sum.shape)
        p=dis_class/dis_sum
        #print('p_new',p)
        return p
    def generate_mask_new(self,centers,test_emb,target,feature,feature_center):
        '''
        centers = [B,D], B centers with dim=D
        test_emb = [num_test,D]
        target = [num_test,1]
        '''
        p=self.compute_p(centers,test_emb,target)
        mask = torch.autograd.grad(p, feature, grad_outputs=torch.ones_like(p), retain_graph=True)[0]

        mask = self.vari_sigmoid(mask,10)
        mask_center = torch.autograd.grad(p, feature_center, grad_outputs=torch.ones_like(p), retain_graph=False)[0]
        mask_center = self.vari_sigmoid(mask_center,10)
        return torch.cat((mask_center,mask),0)

    def generate_centers(self,embed_a):
        num_center = embed_a.shape[0]
        num_spk = int(num_center/self.center_num)
        centers=[]
        for i in range(num_spk):
            embed=torch.mean(embed_a[i*self.center_num:(i+1)*self.center_num,:],dim=0)
            centers.append(embed)
        centers=torch.stack(centers).squeeze().requires_grad_()
        return centers

    def layer_CAM(self,input,target_spk=None, mask=None):
        relu = nn.ReLU()
        score,feature,mel,target = self.auxl(input, target_spk, 'loss', mask)
        self.auxl.zero_grad()
        yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), create_graph=True, retain_graph=True)[0]
        yb = relu(yb)*feature
        yb=torch.sum(yb,1)
        yb_norm = yb/(torch.amax(yb,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)+1e-6)

        return yb_norm,mel,target
 
    def weight_mse(self, pre, clean):
        weight = torch.sum(clean,dim=-1).unsqueeze(-1)
        m = nn.Softmax(dim=1)
        weight = m(weight)
        mse = torch.sum(weight*torch.pow(clean-pre,2))
        return mse

    def train_epoch(self, epoch, weight, device, loader_size, scheduler):
        relu = nn.ReLU()
        running_cos, running_score, running_n, running_tpr, running_tnr, running_mse, running_tht = 0,0,0,0,0,0,0
        i_batch,diff_batch = 0,0
        
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            cur_iter = (epoch - 1) * loader_size + i_batch
            scheduler.step(cur_iter)
            # Load data from trainloader
            clean, noisy, target, noise = sample_batched
            #print(Xb.device)
            _, B, _, W = clean.shape
            clean = clean.reshape(B,W).cuda()
            target = target.squeeze().cuda()
            noise = noise.reshape(B,W).cuda() 
            noisy = noisy.reshape(B,W).cuda()
            mask = self.model(noisy)
            SaM_c,mel_c,target = self.layer_CAM(clean)
            SaM_pre,mel_pre,_ = self.layer_CAM(noisy,target,mask)
            #SaM_n, mel_n, _ = self.layer_CAM(noise, target)
            '''
            for i in range(8):
                
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

                max = torch.max(mel_c[i])
                min = torch.min(mel_c[i])
                librosa.display.specshow(mel_c[i].squeeze().detach().cpu().numpy(), x_axis=None, ax=ax[0])
                librosa.display.specshow(mel_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[1] ,vmin=min,vmax=max)
                #img2=librosa.display.specshow(mel_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[2] ,vmin=min,vmax=max)
                #fig.colorbar(img2, ax=ax)
                plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'mel.png')
                plt.close()
                
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                max = torch.max(SaM_c[i])
                min = torch.min(SaM_c[i])                
                librosa.display.specshow(SaM_c[i].squeeze().detach().cpu().numpy(), x_axis=None, ax=ax[0])
                img2 = librosa.display.specshow(SaM_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[1] ,vmin=min,vmax=max)
                #librosa.display.specshow(SaM_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[2] ,vmin=min,vmax=max)
                #fig.colorbar(img2, ax=ax)
                plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'sam.png')
                plt.close()

                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                max = torch.max(norm_c[i])
                min = torch.min(norm_c[i])                
                librosa.display.specshow(norm_c[i].squeeze().detach().cpu().numpy(), x_axis=None, ax=ax[0])
                img2 = librosa.display.specshow(norm_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[1] ,vmin=min,vmax=max)
                #librosa.display.specshow(SaM_pre[i].detach().cpu().numpy(),x_axis=None, ax=ax[2] ,vmin=min,vmax=max)
                #fig.colorbar(img2, ax=ax)
                plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'norm.png')
                plt.close()

            
            quit()
            '''
            #logits = logits.reshape(logits.shape[0],-1)
            #target_mask = target_mask.reshape(target_mask.shape[0],-1)
            mse_loss = self.weight_mse(SaM_pre, SaM_c.detach())/B
            train_loss = mse_loss
            if torch.isnan(train_loss) or torch.isinf(train_loss):
                torch.save(clean, '/data_a11/mayi/project/SIP/IRM/exp/debug/clean.pt')
                torch.save(noisy,'/data_a11/mayi/project/SIP/IRM/exp/debug/noisy.pt')
                torch.save(target, '/data_a11/mayi/project/SIP/IRM/exp/debug/target.pt')
                state_dict = self.model.state_dict()
                torch.save(state_dict,'/data_a11/mayi/project/SIP/IRM/exp/debug/model.pt')
            #running_cos += self.weight*torch.mean(1-self.cos(logits,target_mask)).item()
            running_mse += mse_loss.item()
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
                self.logger.info("Loss of minibatch {}-th/{}: {}, lr:{}".format(i_batch, len(self.train_dataloader), train_loss.item(), self.optimizer.param_groups[0]['lr']))
        #if epoch%10 ==0:
            #self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*0.8
        #if device==0:
            #if epoch%10 ==0:
                #torch.save(self.model, f"{self.PROJECT_DIR}/models/model_{epoch}.pt")
        if device ==0:
            #ave_cos_loss = running_cos / i_batch
            ave_mse_loss = running_mse / i_batch
            #self.writer.add_scalar('Train/cos', ave_cos_loss, epoch)
            self.writer.add_scalar('Train/mse', ave_mse_loss, epoch)

            self.logger.info("Epoch:{}".format(epoch)) 
            self.logger.info("*" * 50)
    

    def l2_norm(self, pre):
        return torch.mean(torch.norm(pre,p=2,dim=(-1,-2)))
    def set_models_to_eval_mode(self):
        self.model.eval()

    def validation_epoch(self, epoch, weight, device):
        start_time = time.time()
        running_cos, running_score, running_n, running_mse, running_diff_loss, running_tnr_loss, running_tpr_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        i_batch,diff_batch = 0,0
        relu = nn.ReLU()
        
        
        for sample_batched in tqdm(self.validation_dataloader):
            clean, noisy, target, noise = sample_batched
            target = target.squeeze().cuda()
            _, B, _, W = clean.shape
            clean = clean.reshape(B,W).cuda()
            noise = noise.reshape(B,W).cuda()
            noisy = noisy.reshape(B,W).cuda()
            mask = self.model(noisy)
            
            SaM_c,mel_c,target = self.layer_CAM(clean)
            SaM_pre,mel_pre,_ = self.layer_CAM(noisy,target,mask)
            #SaM_n, mel_n, _ = self.layer_CAM(noise, target)
            mse_loss = self.weight_mse(SaM_pre, SaM_c.detach())/B
            #running_cos += self.weight*torch.mean(1-self.cos(logits,target_mask)).item()
            running_mse += mse_loss.item()
            i_batch += 1
            torch.cuda.empty_cache()
        if device==0:    
            #ave_cos_loss = running_cos / i_batch
            ave_mse_loss = running_mse / i_batch
            end_time = time.time()
            #self.writer.add_scalar('Validation/cos', ave_cos_loss, epoch)
            self.writer.add_scalar('Validation/mse', ave_mse_loss, epoch)
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
