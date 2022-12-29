'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''
import matplotlib.pyplot as plt
import math, torch, torchaudio, librosa.display,numpy
import torch.nn as nn
import torch.nn.functional as F
from Models.models import BLSTM
from Models.models import PreEmphasis

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
                 out_features=5994,
                 scale=32.0,
                 margin=0,
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
    def forward(self, input, label, mode='score'):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        #return cosine
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

        
        if mode=='score':

            return output
        else:
            return cosine*self.scale
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
                 pooling_func='TSTP',
                 dur=4):
        super(Speaker_resnet, self).__init__()
        block = BasicBlock
        self.dur=dur
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = False
        self.Spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2.0)
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1)
        self.pre =  PreEmphasis()
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

    def _clip_point(self, frame_len):
        clip_length = 100*self.dur
        start_point = 0
        end_point = frame_len
        if frame_len>=clip_length:
            start_point = numpy.random.randint(0, frame_len - clip_length + 1)
            end_point = start_point + clip_length
        else:
            print('bug exists in clip point')
            quit()
        return start_point, end_point

    def forward(self, x, targets=None, mode='feature', mask=None):
        if mode != 'score':
            with torch.no_grad():
                x = self.pre(x)
                x = self.Spec(x)
                frame_len = x.shape[-1]
                start, end = self._clip_point(frame_len)
                x = x[:,:,start:end]
                x = (self.Mel_scale(x)+1e-6).log()
                if mask is not None:
                    x = x+(mask+1).log()
            #x = (self.Spec(x)+1e-8)
            #x = (self.Mel_scale(x)+1e-8).log()
        #print(x.shape) #[128,80,1002]
        feature = x
        x = feature - torch.mean(feature, dim=-1, keepdim=True)
        #x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        #print('out.shape:',out.shape) #[B,32,80,T]
        out1 = self.layer1(out)
        #print('layer1 shape:',out.shape) #[B,32,80,T]
        out2 = self.layer2(out1)
        #print('layer2 shape:',out.shape) #[B,64,40,T/2]
        out3 = self.layer3(out2)
        #print('layer3 shape:',out.shape)#[B,128,20,T/4]
        out4 = self.layer4(out3)
        #print('layer4 shape:',out.shape)#[B,256,10,T/8]
        #frame = out4.requires_grad_()
        stats = self.pool(out4)
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
        elif mode == 'encoder':
            return [out1,out2,out3,out4,embed_a],feature
        elif mode == 'reference':
            return embed_a
        elif mode in ['score','loss']:
            score = self.projection(embed_a, targets, mode)
            result = torch.gather(score,1,targets.unsqueeze(1).long()).squeeze()
            if mode == 'score':
                return result
            else:
                return score

class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def CNNBlock(in_channels, out_channels,
                 kernel_size=3, layers=1, stride=1,
                 follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):

        assert layers > 0 and kernel_size%2 and stride>0
        current_channels = in_channels
        _modules = []
        for layer in range(layers):
            _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer==0 else 1, padding=int(kernel_size/2), bias=not follow_with_bn))
            current_channels = out_channels
            if follow_with_bn:
                _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
            if activation_fn is not None:
                _modules.append(activation_fn())
        return nn.Sequential(*_modules)

def SubpixelUpsampler(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False), follow_with_bn=True):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),#[B,1024,4,4]
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)

class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels,passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels,out_channels=out_channels)
        self.follow_up = Block(out_channels+passthrough_channels,out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out,passthrough), 1)
        return self.follow_up(out)

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.uplayer4 = UpSampleBlock(in_channels=256,out_channels=128,passthrough_channels=128)
        self.uplayer3 = UpSampleBlock(in_channels=128,out_channels=64,passthrough_channels=64)
        self.uplayer2 = UpSampleBlock(in_channels=64,out_channels=32,passthrough_channels=32)
        self.saliency_chans = nn.Conv2d(32,2,kernel_size=1,bias=False)
    def vari_sigmoid(self,x,a):
        return 1/(1+(-a*x).exp())

    def forward(self,encoder_out):
        em = encoder_out[-1]
        scale4 = encoder_out[-2]
        act = torch.sum(scale4*em.view(-1, 256, 1, 1), 1, keepdim=True)
        th = torch.sigmoid(act)
        scale4 = scale4*th

        upsample3 = self.uplayer4(scale4, encoder_out[-3])
        upsample2 = self.uplayer3(upsample3, encoder_out[-4])
        upsample1 = self.uplayer2(upsample2, encoder_out[-5])
        saliency_chans = self.saliency_chans(upsample1)
        a = torch.abs(saliency_chans[:,0,:,:])
        b = torch.abs(saliency_chans[:,1,:,:])
        return a/(a+b+1e-6)

class multi_TDNN(nn.Module):
    
    def __init__(self,dur):
        super(multi_TDNN, self).__init__()
        
        self.decoder = decoder()

        self.speaker = Speaker_resnet(dur=dur)
        #for name in self.speaker.parameters():
            #print(name)

        path = "exp/resnet_5994.pt"
        checkpoint = torch.load(path)
        self.speaker.load_state_dict(checkpoint, strict=False)

        for p in self.speaker.parameters():
            #if 'projection' in name:
                #print('param',param)
            p.requires_grad = False
        self.speaker.eval()
    def forward(self,input):
        self.speaker.eval()
        #for name, param in self.speaker.parameters():
            #print(name)
        #print(next(self.speaker.parameters()).device)
            #print('input:{},speaker:{}'.format(input.device,i.device))
        
        encoder_out,feature = self.speaker(input,mode='encoder')
        mask = self.decoder(encoder_out)
        return mask,feature
