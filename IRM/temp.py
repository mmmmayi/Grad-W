import torch
import numpy as np
def accuracy(output, target):
    print('output:',output)
    maxk = 1
    batch_size, T,F = target.shape
    _, pred = output.topk(maxk, 1, True, True)
    pred = torch.sum(pred,1)
    print('pred:',pred)
    pred = pred.reshape(-1)    
    p_idx = torch.nonzero(pred).squeeze()
    print('p_idx:',p_idx)
    t_idx = torch.nonzero(target.reshape(-1)).squeeze()
    print('t_idx:',t_idx)
    tpr = len(np.intersect1d(p_idx,t_idx))/t_idx.numel()
    print('tpr:',tpr)
    n_idx = (pred == 0).nonzero().squeeze()
    print('n_idx:',n_idx)
    f_idx = (target.reshape(-1) == 0).nonzero().squeeze()
    print('f_idx:',f_idx)
    tnr = len(np.intersect1d(n_idx,f_idx))/f_idx.numel()
    print('tnr:',tnr)

    #for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #res.append(correct_k.mul_(100.0 / batch_size))
	

if __name__ =='__main__':
    output = torch.randn(2,2,3,3)
    target = torch.tensor([[[0,1,1],[1,0,0],[1,0,1]],[[0,1,1],[1,0,0],[1,0,1]]])
    accuracy(output,target)

