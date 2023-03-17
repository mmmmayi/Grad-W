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
            local_rank,w_p,w_n,w_hn):
        # Init device
        self.config = config
        self.weight = w_p
        self.w_n = w_n
        self.w_hn = w_hn
        # Dataloaders
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl  ##to check

        # Model Params
        self.optimizer = optimizer
        self.ratio = ratio
        self.cos = loss_fn[0]
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

    def generate_mask(self,embed_a,test_emb,feature):
        num_center = embed_a.shape[0]
        c1 = torch.mean(embed_a[:int(num_center/2),:],dim=0).unsqueeze(0).detach()
        c2 = torch.mean(embed_a[int(num_center/2):,:],dim=0).unsqueeze(0).detach()
        test_num = int(test_emb.shape[0]/2)
        c1 = c1.repeat(test_num,1)
        c2 = c2.repeat(test_num,1)
        sim_center = torch.cat((c1,c2),dim=0)
        dissim_center = torch.cat((c2,c1),dim=0)

        loss = self.cos_emb(test_emb,sim_center,torch.tensor([1]*test_emb.shape[0]).cuda())+self.cos_emb(test_emb,dissim_center,torch.tensor([-1]*test_emb.shape[0]).cuda())
        mask = torch.autograd.grad(loss, feature, grad_outputs=torch.ones_like(loss), retain_graph=False)[0]
        return 0-mask,sim_center,dissim_center

    def train_epoch(self, epoch, weight, device, loader_size, scheduler):
        relu = nn.ReLU()
        running_cos, running_score, running_n, running_tpr, running_tnr, running_mse, running_tht = 0,0,0,0,0,0,0
        i_batch,diff_batch = 0,0
        
        self.train_dataloader.dataset.shuffle(epoch)
        for sample_batched in self.train_dataloader:
            cur_iter = (epoch - 1) * loader_size + i_batch
            scheduler.step(cur_iter)
            # Load data from trainloader
            center, test = sample_batched
            #print(Xb.device)
            _, num_center, _, W = center.shape
            center = center.reshape(num_center,W).cuda()
            _, num_test, _, W = test.shape
            test = test.reshape(num_test,W).cuda()
            
            logits, feature_n, points = self.model(test)
            embed_a = self.auxl(center,mode='reference')
            test_emb, feature = self.auxl(test,None,'score',None,points)
            self.auxl.zero_grad()
            target_mask,sim_center,dissim_center = self.generate_mask(embed_a,test_emb,feature)
            #feature = feature.detach().requires_grad_()
            '''
            if device==0:
                for i in range(4):
                    #mask_ = np.sort(yb[i,:,:].detach().cpu().squeeze().numpy(),axis=None).squeeze()
                    #x = np.arange(len(mask_))
                    #plt.plot(x, mask_, color='blue', label='predict')
                    #plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'sort.png')

                    #plt.close()
 
                    #temp2 = self.vari_ReLU(yb,self.ratio,device) 
                    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
                    max=torch.max(feature[i,:,:])
                    min = torch.min(feature[i,:,:]) 
                    print('max',torch.max((logits[i,:,:])))
                    librosa.display.specshow(feature_n[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0],vmax=max,vmin=min)
                    librosa.display.specshow(feature[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
#                    img = librosa.display.specshow(mask[i,:,:].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
                    img=librosa.display.specshow((logits[i,:,:]+0.5).log().detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
                    librosa.display.specshow((feature[i,:,:]-feature_n[i,:,:]).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[3])

                    #print(weight[i,:,:])
                    fig.colorbar(img, ax=ax)

                    #img = librosa.display.specshow(yb[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
                    plt.savefig('/data_a11/mayi/project/SIP/IRM/exp/debug/'+str(i)+'varied.png')
                    plt.close()
                quit() 
            '''
            #inverse_mask = 1-mask
            
            #mse = self.mse(mask, SaM.float())
            test_emb, feature = self.auxl(test,None,'score',logits,points)
            loss_drct = self.cos_emb(test_emb,sim_center,torch.tensor([1]*test_emb.shape[0]).cuda())+self.cos_emb(test_emb,dissim_center,torch.tensor([-1]*test_emb.shape[0]).cuda())
            logits = logits.reshape(logits.shape[0],-1)
            target_mask = target_mask.reshape(target_mask.shape[0],-1)
            train_loss = self.weight*torch.mean(1-self.cos(logits,target_mask))+loss_drct
            if torch.isnan(train_loss) or torch.isinf(train_loss):
                torch.save(center, '/data_a11/mayi/project/SIP/IRM/exp/debug/center.pt')
                torch.save(test,'/data_a11/mayi/project/SIP/IRM/exp/debug/test.pt')
                state_dict = self.model.state_dict()
                torch.save(state_dict,'/data_a11/mayi/project/SIP/IRM/exp/debug/model.pt')
                quit()
            running_cos += self.weight*torch.mean(1-self.cos(logits,target_mask)).item()
            running_score += loss_drct.item()
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
            ave_cos_loss = running_cos / i_batch
            ave_score_loss = running_score / i_batch
            self.writer.add_scalar('Train/cos', ave_cos_loss, epoch)
            self.writer.add_scalar('Train/score', ave_score_loss, epoch)

            self.logger.info("Epoch:{}".format(epoch)) 
            self.logger.info("*" * 50)
    

    
    def set_models_to_eval_mode(self):
        self.model.eval()

    def validation_epoch(self, epoch, weight, device):
        start_time = time.time()
        running_cos, running_score, running_n, running_mse_loss, running_diff_loss, running_tnr_loss, running_tpr_loss  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        i_batch,diff_batch = 0,0
        relu = nn.ReLU()
        
        
        for sample_batched in tqdm(self.validation_dataloader):
            center, test = sample_batched
            _, num_center, _, W = center.shape
            center = center.reshape(num_center,W).cuda()
            _, num_test, _, W = test.shape
            test = test.reshape(num_test,W).cuda()

            logits, feature_n, points = self.model(test)
            embed_a = self.auxl(center,mode='reference')
            test_emb, feature = self.auxl(test,None,'score',None,points)

            self.auxl.zero_grad()
            target_mask,sim_center,dissim_center = self.generate_mask(embed_a,test_emb,feature)
            test_emb, feature = self.auxl(test,None,'score',logits,points)
            loss_drct = self.cos_emb(test_emb,sim_center,torch.tensor([1]*test_emb.shape[0]).cuda())+self.cos_emb(test_emb,dissim_center,torch.tensor([-1]*test_emb.shape[0]).cuda())
            logits = logits.reshape(logits.shape[0],-1)
            target_mask = target_mask.reshape(logits.shape[0],-1)

            running_cos += self.weight*torch.mean(1-self.cos(logits,target_mask)).item()
            running_score += loss_drct.item()
            i_batch += 1
        if device==0:    
            ave_cos_loss = running_cos / i_batch
            ave_score_loss = running_score / i_batch
            end_time = time.time()
            self.writer.add_scalar('Validation/cos', ave_cos_loss, epoch)
            self.writer.add_scalar('Validation/score', ave_score_loss, epoch)
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
