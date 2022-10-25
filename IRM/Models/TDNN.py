'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''
import matplotlib.pyplot as plt
import math, torch, torchaudio, librosa.display
import torch.nn as nn
import torch.nn.functional as F
from Models.models import BLSTM

class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 

class AAMsoftmax(nn.Module):
    def __init__(self, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.load('../lst/loss')
        self.weight.requires_grad = False
        self.ce = nn.CrossEntropyLoss()
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
    
class SEModule_2d(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule_2d, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck_2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck_2d, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv2d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.InstanceNorm2d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.InstanceNorm2d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv2d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.InstanceNorm2d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule_2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 


class Speaker_TDNN(nn.Module):

    def __init__(self, C=1024):

        super(Speaker_TDNN, self).__init__()

        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
                nn.Conv1d(4608, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Tanh(), # I add this layer
                nn.Conv1d(256, 1536, kernel_size=1),
                nn.Softmax(dim=2),
                )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.speaker_loss = AAMsoftmax( m = 0.2, s = 30).cuda()
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1).cuda()

    def forward(self, x, target_spk=None, mode='embedding'):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x0 = x
        x1 = self.layer1(x)

        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))

        x = self.relu(x)
        x_frame = x
        t = x.size()[-1]
        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )
        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        if mode=='feature':
            return x0, x1, x2, x3, x_frame
        elif mode == 'score':
            score = self.speaker_loss.forward(x,target_spk)
            result = torch.gather(score,1,target_spk.unsqueeze(1).long()).squeeze()
 
            return result
        return x

class multi_TDNN(nn.Module):
    
    def __init__(self, configs):
        super(multi_TDNN, self).__init__()
        
        self.module = BLSTM(configs)
        path = '../exps/pretrain.model'
        loaded_state = torch.load(path)
        self.speaker = Speaker_TDNN()
        self_state = self.speaker.state_dict()
        for name, param in loaded_state.items():
            if name not in self_state:
                name = name.replace("speaker_encoder.","")
                if name not in self_state:
                    continue
            self_state[name].copy_(param)
        for p in self.speaker.parameters():
            p.requires_grad = False
        self.speaker.eval()
    def forward(self,input_spk, input, target_spk=None):
        self.speaker.eval()
       
        self.mask = []
        with torch.no_grad():
            x0, x1, x2, x3, x_frame = self.speaker(input_spk, target_spk, mode='feature')
        mask = self.module(input, x_frame) #range of mask is [0,1]

        return mask
