import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as nnf
def accuracy(output, weight):
    maxk = 1
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = torch.sum(pred,1)
    print('pred',pred)
    pred = pred.reshape(-1)
    
    p_idx = torch.nonzero(pred).squeeze()
    t_idx = (weight.reshape(-1) == 0.95).nonzero().squeeze()
    tpr = len(np.intersect1d(p_idx.detach().cpu(),t_idx.detach().cpu()))/t_idx.numel()
    n_idx = (pred == 0).nonzero().squeeze()
    print('n_idx',n_idx)
    f_idx = (weight.reshape(-1) == 0.5).nonzero().squeeze()
    print('f_idx',f_idx)
    tnr = len(np.intersect1d(n_idx.detach().cpu(),f_idx.detach().cpu()))/f_idx.numel()
    return tpr, tnr


    #fo k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #res.append(correct_k.mul_(100.0 / batch_size))
	
def ce(output,target,weight):
    B,C,T,F = output.size()
    output = output.reshape(B,C,T*F)
    target = target.reshape(B,1,T*F)
    weight = weight.reshape(B,T*F)
    output = torch.exp(output)
    output_sum = torch.sum(output,1)
    output_class = torch.gather(output,1,target).squeeze()
    
    ce = -torch.log(output_class/output_sum)*weight
    
    
    result = torch.sum(ce)/torch.sum(weight)
    print(result)
    ce = ce.reshape(-1)
    weight = weight.reshape(-1)
    idx_p = (weight==0.95).nonzero().squeeze()
    idx_b = (weight==0.5).nonzero().squeeze()
    idx_n = (weight==0.05).nonzero().squeeze()
    
    ce_p = torch.gather(ce,0,idx_p)
    ce_b = torch.gather(ce,0,idx_b)
    ce_n = torch.gather(ce,0,idx_n)
    result = torch.sum(ce_p)/torch.sum(weight)+torch.sum(ce_b)/torch.sum(weight)+torch.sum(ce_n)/torch.sum(weight)
    print(result)

    


if __name__ =='__main__':
    output = torch.randn(2,2,4,5)
    target = torch.tensor([[[0,1,1],[1,0,0],[1,0,1]],[[0,1,1],[1,0,0],[1,0,1]]])
    weight = torch.FloatTensor([[[0.05,0.95,0.95],[0.95,0.05,0.05],[0.95,0.05,0.95]],[[0.05,0.95,0.95],[0.95,0.05,0.05],[0.95,0.05,0.95]]])
    #ce(output,target, weight)
    #loss = nn.CrossEntropyLoss(weight = torch.FloatTensor([0.05,0.95]))
    #print(loss(output,target))
    
    mask = torch.randn(2,1,4,5)
    mask = torch.where(mask>0.8,1,0)
    box = torch.ones((3, 3), dtype=mask.dtype, requires_grad=False) 
    box = box[None, None, ...].repeat(1, 1, 1, 1)
    mask_edit = nnf.conv2d(mask, box, padding=1,groups=mask.size(1))
    mask_edit = torch.where(mask_edit>0,0.5,0.05)
    weight = torch.where(mask==1,0.95,mask_edit)
    
    #print('weight',weight)
    #print(accuracy(torch.randn(2,2,4,5),weight))
    #print(weight)
    ce(output,mask,weight)
    

