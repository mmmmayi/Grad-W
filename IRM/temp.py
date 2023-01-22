import torch
def accuracy(output, target):
    print('output:',output)
    maxk = 1
    batch_size, T,F = target.shape
    _, pred = output.topk(maxk, 1, True, True)
    pred = torch.sum(pred,1)
    print('pred:',pred)
    print(pred.shape)
    correct = pred.eq(target)
    print('corrrect:',correct)
    correct = correct.float().sum()
    print(correct)
    res = correct/(batch_size*T*F)
    print(res)
    #for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        #res.append(correct_k.mul_(100.0 / batch_size))
	
    return res

if __name__ =='__main__':
    output = torch.randn(2,2,3,3)
    target = torch.tensor([[[0,1,1],[1,0,0],[1,0,1]],[[0,1,1],[1,0,0],[1,0,1]]])
    accuracy(output,target)

