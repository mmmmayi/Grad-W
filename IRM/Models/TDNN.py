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
            return x_frame
        elif mode == 'score':
            score = self.speaker_loss.forward(x,target_spk)
            result = torch.gather(score,1,target_spk.unsqueeze(1).long()).squeeze()
 
            return result
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
        """
    def __init__(self,
                 in_features=256,
                 out_features=17982,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={},
                  margin={}, easy_margin={}'''.format(self.in_features,
                                                      self.out_features,
                                                      self.scale, self.margin,
                                                      self.easy_margin)

class TSTP(nn.Module):
    """
    Temporal statistics pooling, concatenate mean and std, which is used in
    x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    def __init__(self, **kwargs):
        super(TSTP, self).__init__()

    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats


class Speaker_resnet(nn.Module):
    def __init__(self,
                 num_blocks=[3,4,6,3],
                 m_channels=32,
                 feat_dim=80,
                 embed_dim=256,
                 pooling_func='TSTP'):
        super(Speaker_resnet, self).__init__()
        block = BasicBlock
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = False
        self.Spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2.0)
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1)
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)

        self.n_stats = 1 if pooling_func == 'TAP' or pooling_func == "TSDP" else 2
        self.pool = TSTP()
        self.seg_1 = nn.Linear(self.stats_dim * block.expansion * self.n_stats,
                               embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, mode='feature'):
            #x = (self.Spec(x)+1e-8)
            #x = (self.Mel_scale(x)+1e-8).log()
            #print(x.shape) #[128,80,1002]
        x = x - torch.mean(x, dim=-1, keepdim=True)
        #x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)

        x = x.unsqueeze_(1)
        print('x shape:',x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        print('out.shape:',out.shape)
        out = self.layer1(out)
        print('layer1 shape:',out.shape)
        out = self.layer2(out)
        print('layer2 shape:',out.shape)
        out = self.layer3(out)
        print('layer3 shape:',out.shape)
        out = self.layer4(out)
        print('layer4 shape:',out.shape)
        quit()
        frame = out
        #print(frame.shape)#[B,256,10,51]
        stats = self.pool(out)

        embed_a = self.seg_1(stats)
        '''
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_a, embed_b
        else:
            return x, embed_a
        '''
        if mode=='feature':
            return frame
        elif mode == 'score':
            score = self.projection(embed_a, targets)
            result = torch.gather(score,1,targets.unsqueeze(1).long()).squeeze()
            return result







class multi_TDNN(nn.Module):
    
    def __init__(self, configs, encoder='ECAPA'):
        super(multi_TDNN, self).__init__()
        
        self.module = BLSTM(configs)
        if encoder == 'ECAPA':
            path = '../exp/pretrain.model'
            loaded_state = torch.load(path)
            self.speaker = Speaker_TDNN()
            self_state = self.speaker.state_dict()
            for name, param in loaded_state.items():
                if name not in self_state:
                    name = name.replace("speaker_encoder.","")
                    if name not in self_state:
                        continue
                self_state[name].copy_(param)

        elif encoder == 'resnet':
            self.speaker = Speaker_resnet()
            path = "exp/resnet.pt"
            checkpoint = torch.load(path)
            self.speaker.load_state_dict(checkpoint, strict=False)

        for p in self.speaker.parameters():
            p.requires_grad = False
        self.speaker.eval()
    def forward(self,input_spk, input, target_spk=None):
        self.speaker.eval()
       
        self.mask = []
        with torch.no_grad():
            x_frame = self.speaker(input_spk, target_spk, mode='feature')
        mask = self.module(input, x_frame) #range of mask is [0,1]

        return mask
