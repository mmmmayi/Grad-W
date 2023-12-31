import os, torchaudio, soundfile,csv,math
from pystoi import stoi
from pypesq import pesq 
import tqdm
import torch
import librosa
import librosa.display
import numpy as np
import tools.utils as utils
from Models.TDNN import  multi_TDNN
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
from Models.models import PreEmphasis
from tools.metrics import compute_PESQ, compute_segsnr
import torch.nn as nn
from Models.TDNN import Speaker_resnet, ArcMarginProduct
from tools.metrics import compute_PESQ, compute_segsnr,ComputeErrorRates,ComputeMinDcf,tuneThresholdfromScore
import torch.nn.functional as F
class IRMApplier():
    def __init__(self,
            project_dir, test_set_name, sampling_rate,
            model_path, applier_dl, mode):
        self.mode = mode
        self.model_path = model_path
        self.applier_dataloader = applier_dl
        self.sampling_rate = sampling_rate
        self.test_set_name = test_set_name
        self.pre = PreEmphasis()
        self.InvSpec = torchaudio.transforms.InverseSpectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window).cuda()
        self.cplx = Spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=None).cuda()

        self.Spec = torchaudio.transforms.Spectrogram(n_fft=512, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2).cuda()
        self.Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,512//2+1).cuda()
        self.MelSpec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80).cuda()
        # Init Output folder
        self.PROJECT_DIR = project_dir
        
        self.mean = torch.load("exp/resnet_mean.pt").float().cuda()
        self.std = torch.load("exp/resnet_mean.pt").float().cuda()
        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Load model
        self.model = multi_TDNN().cuda()
        checkpoint = torch.load(model_path)
        '''
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        '''
        self.model.load_state_dict(checkpoint)
        self.loss = nn.MSELoss()
        self.model.eval()
        self.auxl = Speaker_resnet().cuda()
        projection = ArcMarginProduct().cuda()
        self.auxl.add_module("projection", projection)
        path = 'exp/resnet_babble.pt'
        checkpoint = torch.load(path)
        self.auxl.load_state_dict(checkpoint, strict=False)
        self.labels = torch.load('../lst/resnet_speaker.pt')

        for p in self.auxl.parameters():
            p.requires_grad = False
        self.auxl.eval()


    def restore(self):
        torch.save(self.model.state_dict(),self.model_path.replace('.pt','_para.pt'))
        
    def lps2wav(self, cam, wave):
        cam = torch.sqrt(cam)
        orin_wav = self.cplx(wave)
        pre_real = orin_wav.real*cam
        pre_imag = orin_wav.imag*cam
        pre = torch.complex(pre_real, pre_imag) 
        x= self.InvSpec(pre, length = wave.size()[-1])

        return x
    def compute_SISNR(self, ref_sig, out_sig, eps=1e-8):
        """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
        Args:
            ref_sig: numpy.ndarray, [T]
            out_sig: numpy.ndarray, [T]
  
        """
        assert len(ref_sig) == len(out_sig)
        ref_sig = ref_sig - np.mean(ref_sig)
        out_sig = out_sig - np.mean(out_sig)
        ref_energy = np.sum(ref_sig ** 2) + eps
        proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
        noise = out_sig - proj
        ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
        sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
        return sisnr

    def compute_segsnr(self, clean_signal, noisy_signal, sr=16000, frame:int=0, overlap:int=0):
        MIN_ALLOWED_SNR = -10.0
        MAX_ALLOWED_SNR = +35.0
        EPSILON = 0.0000001

        noisy_signal = noisy_signal[:]
        clean_signal = clean_signal[:]

        min_len = min(len(clean_signal), len(noisy_signal))

        original = clean_signal[:min_len]
        degraded = noisy_signal[:min_len]

        snr = 10.0 * np.log10(np.sum(np.square(original)) / np.sum(np.square(original - degraded)))

        if frame == 0:
            frame_size_smpls = int(np.floor(2.0 * sr / 100.0))
        else:
            frame_size_smpls = frame

        if overlap == 0:
            overlap_perc = 50.0
        else:
            overlap_perc = overlap

        overlap_size_smpls = int(np.floor(frame_size_smpls * overlap_perc / 100.0))
        step_size = frame_size_smpls - overlap_size_smpls

        window = np.hanning(frame_size_smpls)

        n = int(min_len / step_size - (frame_size_smpls / step_size))

        seg_snrs = []

        ind = 0

        for i in range(n):
            original_frame = original[ind:ind + frame_size_smpls] * window
            degraded_frame = degraded[ind:ind + frame_size_smpls] * window
 
            speech_power = np.sum(np.square(original_frame))
            noise_power = np.sum(np.square(original_frame - degraded_frame))

            seg_snr = 10.0 * np.log10(speech_power / (noise_power + EPSILON) + EPSILON)
            seg_snr = np.max([seg_snr, MIN_ALLOWED_SNR])
            seg_snr = np.min([seg_snr, MAX_ALLOWED_SNR])

            seg_snrs.append(seg_snr)

            ind += step_size

        seg_snr = np.mean(seg_snrs)
        return snr, seg_snr
    def quality(self):
        SISNR, SISNR_before, PESQ, PESQ_before, STOI, STOI_before, num= 0,0,0,0,0,0,0
        score = []
        total_files = len(self.applier_dataloader)
        for sample_batched in tqdm(self.applier_dataloader):
            Xb, clean = sample_batched
            assert Xb.size()[0] == 1, "The batch size of validation dataloader must be 1."
            Xb = Xb.reshape(1,Xb.size()[-1]).cuda()
            clean = clean.reshape(1,Xb.size()[-1]).cuda()
            noisy = Xb
            speech = clean
            clean = self.pre(clean)
            clean = (self.Spec(clean)+1e-8).log()
            Xb = self.pre(Xb)
            Xb = (self.Spec(Xb)+1e-8).log()
            mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()

            Xb_input = (Xb-self.mean)/self.std
            cam = self.model(mel_n, Xb_input)
            enhanced_spec = (cam+1e-8).log()+Xb

            #if not os.path.exists(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2])):
                #os.makedirs(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2])) 
            waveform = self.lps2wav(cam, noisy).detach().cpu().numpy().squeeze()

            enhanced_signal = waveform
            noisy = noisy.detach().cpu().numpy().squeeze()
            speech = speech.detach().cpu().numpy().squeeze()
            try: 
                snr, seg_snr = self.compute_segsnr(speech,enhanced_signal, sr=16000)
                s = stoi(speech, enhanced_signal, 16000, extended=False)
                p = pesq(speech,enhanced_signal,16000)
                sisnr = self.compute_SISNR(speech,enhanced_signal)

                snr_before, _ = self.compute_segsnr(speech, noisy,sr=16000)
                s_before = stoi(speech, noisy, 16000, extended=False)
                p_before = pesq(speech, noisy,16000)
                sisnr_before = self.compute_SISNR(speech, noisy)
            except:
                
                print(enhanced_signal)
                    
            SISNR+=sisnr
            SISNR_before+=sisnr_before
            PESQ+=p
            PESQ_before+=p_before
            STOI+=s
            STOI_before+=s_before
            num+=1
        print('SISNR:{}, SISNR_before:{}'.format(str(SISNR/num), str(SISNR_before/num))+'\n')
        print('PESQ:{}, PESQ_before:{}'.format(str(PESQ/num), str(PESQ_before/num))+'\n')
        print('STOI:{}, STOI_before:{}'.format(str(STOI/num), str(STOI_before/num))+'\n')

    def quality_vox1(self):
        num_0, num_5, num_10, num_15, num_20 = 1,1,1,1,1
        SISNR, SISNR_before, PESQ, PESQ_before, STOI, STOI_before, num= 0,0,0,0,0,0,0
        SISNR_0, SISNR_before_0 = 0,0
        PESQ_before_0, PESQ_0 = 0,0
        STOI_0, STOI_before_0 = 0,0

        SISNR_5, SISNR_before_5 = 0,0
        PESQ_before_5, PESQ_5 = 0,0
        STOI_5, STOI_before_5 = 0,0

        SISNR_10, SISNR_before_10 = 0,0
        PESQ_before_10, PESQ_10 = 0,0
        STOI_10, STOI_before_10 = 0,0

        SISNR_15, SISNR_before_15 = 0,0
        PESQ_before_15, PESQ_15 = 0,0
        STOI_15, STOI_before_15 = 0,0

        SISNR_20, SISNR_before_20 = 0,0
        PESQ_before_20, PESQ_20 = 0,0
        STOI_20, STOI_before_20 = 0,0
        score = []
        total_files = len(self.applier_dataloader)
        for sample_batched in tqdm.tqdm(self.applier_dataloader):
            Xb, clean = sample_batched
            #info = info[0].split('/')
            #idx = int(info[-3][2:])
            #num = str(idx//200)

            #target_spk = target_spk.reshape(1)
            assert Xb.size()[0] == 1, "The batch size of validation dataloader must be 1."
            Xb = Xb.reshape(1,Xb.size()[-1]).cuda()
            clean = clean.reshape(1,Xb.size()[-1]).cuda()
            noisy = Xb
            speech = clean
            clean = self.pre(clean)
            clean = (self.Spec(clean)+1e-8).log()
            Xb = self.pre(Xb)
            Xb = (self.Spec(Xb)+1e-8).log()
            mel_n = (self.Mel_scale(Xb.exp())+1e-6).log()

            Xb_input = (Xb-self.mean)/self.std
            cam = self.model(mel_n, Xb_input, Xb)
            enhanced_spec = (cam+1e-8).log()+Xb

            #if not os.path.exists(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2])):
                #os.makedirs(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2]))
            waveform = self.lps2wav(cam, noisy).detach().cpu().numpy().squeeze()

            #soundfile.write(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2],info[-1]), waveform, 16000)

            enhanced_signal = waveform
            noisy = noisy.detach().cpu().numpy().squeeze()
            speech = speech.detach().cpu().numpy().squeeze()
            try:
                snr, seg_snr = self.compute_segsnr(speech,enhanced_signal, sr=16000)
                s = stoi(speech, enhanced_signal, 16000, extended=False)
                p = pesq(speech,enhanced_signal,16000)
            #sdr, _, _, _ = bss_eval_sources(clean_signal, enhanced_signal)
                sisnr = self.compute_SISNR(speech,enhanced_signal)

                snr_before, _ = self.compute_segsnr(speech, noisy,sr=16000)
                s_before = stoi(speech, noisy, 16000, extended=False)
                p_before = pesq(speech, noisy,16000)
                sisnr_before = self.compute_SISNR(speech, noisy)
            #sdr_before, _, _, _ = bss_eval_sources(clean_signal,mix_signal)
            except:

                print(enhanced_signal)
            if True:
                if abs(snr_before-0)<1:
                    num_0 = num_0+1

                    SISNR_0 = SISNR_0+sisnr
                    PESQ_0 = PESQ_0+p

                    STOI_0 = STOI_0+s
                    SISNR_before_0 = SISNR_before_0+sisnr_before
                    PESQ_before_0 = PESQ_before_0+p_before

                    STOI_before_0 = STOI_before_0+s_before
                elif abs(snr_before-5)<1:
                    num_5 = num_5+1
                    SISNR_5 = SISNR_5+sisnr

                    PESQ_5 = PESQ_5+p

                    STOI_5 = STOI_5+s
                    SISNR_before_5 = SISNR_before_5+sisnr_before
                    PESQ_before_5 = PESQ_before_5+p_before

                    STOI_before_5 = STOI_before_5+s_before

                elif abs(snr_before-10)<1:
                    num_10 = num_10+1
                    SISNR_10 = SISNR_10+sisnr

                    PESQ_10 = PESQ_10+p

                    STOI_10 = STOI_10+s
                    SISNR_before_10 = SISNR_before_10+sisnr_before
                    PESQ_before_10 = PESQ_before_10+p_before

                    STOI_before_10 = STOI_before_10+s_before

                elif abs(snr_before-15)<1:
                    num_15 = num_15+1
                    SISNR_15 = SISNR_15+sisnr

                    PESQ_15 = PESQ_15+p

                    STOI_15 = STOI_15+s
                    SISNR_before_15 = SISNR_before_15+sisnr_before
                    PESQ_before_15 = PESQ_before_15+p_before

                    STOI_before_15 = STOI_before_15+s_before

                elif abs(snr_before-20)<1:
                    num_20 = num_20+1
                    SISNR_20 = SISNR_20+sisnr

                    PESQ_20 = PESQ_20+p

                    STOI_20 = STOI_20+s
                    SISNR_before_20 = SISNR_before_20+sisnr_before
                    PESQ_before_20 = PESQ_before_20+p_before

                    STOI_before_20 = STOI_before_20+s_before

                else:
                    print('bug exist'+str(snr_before))

            else:
                SISNR+=sisnr
                SISNR_before+=sisnr_before
                PESQ+=p
                PESQ_before+=p_before
                STOI+=s
                STOI_before+=s_before
                num+=1
        score.append({
                    'sisnr_0': SISNR_0/num_0,
                    'sisnr_before_0':SISNR_before_0/num_0,
                    'PESQ_0': PESQ_0/num_0,
                    'PESQ_before_0':  PESQ_before_0/num_0,
                    'STOI_0':STOI_0/num_0,
                    'STOI_before_0':STOI_before_0/num_0,

                    'sisnr_5': SISNR_5/num_5,
                    'sisnr_before_5':SISNR_before_5/num_5,
                    'PESQ_5': PESQ_5/num_5,
                    'PESQ_before_5':  PESQ_before_5/num_5,
                    'STOI_5':STOI_5/num_5,
                    'STOI_before_5':STOI_before_5/num_5,

                    'sisnr_10': SISNR_10/num_10,
                    'sisnr_before_10':SISNR_before_10/num_10,
                    'PESQ_10': PESQ_10/num_10,
                    'PESQ_before_10': PESQ_before_10/num_10,
                    'STOI_10':STOI_10/num_10,
                    'STOI_before_10':STOI_before_10/num_10,

                    'sisnr_15': SISNR_15/num_15,
                    'sisnr_before_15':SISNR_before_15/num_15,
                    'PESQ_15': PESQ_15/num_15,
                    'PESQ_before_15': PESQ_before_15/num_15,
                    'STOI_15':STOI_15/num_15,
                    'STOI_before_15':STOI_before_15/num_15,

                    'sisnr_20': SISNR_20/num_20,
                    'sisnr_before_20':SISNR_before_20/num_20,
                    'PESQ_20': PESQ_20/num_20,
                    'PESQ_before_20': PESQ_before_20/num_20,
                    'STOI_20':STOI_20/num_20,
                    'STOI_before_20':STOI_before_20/num_20


                })
        csv_rows= ['sisnr_0', 'sisnr_before_0', 'PESQ_0', 'PESQ_before_0', 'STOI_0', 'STOI_before_0', 'sisnr_5', 'sisnr_before_5', 'PESQ_5', 'PESQ_before_5', 'STOI_5', 'STOI_before_5', 'sisnr_10', 'sisnr_before_10', 'PESQ_10', 'PESQ_before_10', 'STOI_10', 'STOI_before_10', 'sisnr_15', 'sisnr_before_15', 'PESQ_15', 'PESQ_before_15', 'STOI_15', 'STOI_before_15', 'sisnr_20', 'sisnr_before_20', 'PESQ_20', 'PESQ_before_20', 'STOI_20', 'STOI_before_20']
        if not os.path.exists(self.PROJECT_DIR):
            os.makedirs(self.PROJECT_DIR)
        csv_file = os.path.join(self.PROJECT_DIR,'quality_vox1.csv')
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_rows)
            writer.writeheader()
            for data in score:
                print(data)
                writer.writerow(data)

    def _clip_point(self, frame_len):
        clip_length = 100*4
        start_point = 0
        end_point = frame_len
        if frame_len>=clip_length:
            start_point = np.random.randint(0, frame_len - clip_length + 1)
            end_point = start_point + clip_length
        else:
            print('bug exists in clip point')
            quit()
        return start_point, end_point
    def vari_sigmoid(self,x,a):
        return 1/(1+(-a*x).exp())
    #@torch.no_grad()

    def layer_CAM(self,input,target_spk=None, mask=None):
        relu = nn.ReLU()
        score,feature,mel,target = self.auxl(input, target_spk, 'loss', mask)
        self.auxl.zero_grad()
        yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), create_graph=True, retain_graph=True)[0]
        #if feature_c is None:
            #feature_c = feature
        #yb = relu(yb)*feature_c
        #yb=torch.sum(yb,1)
        return yb,mel,target,feature

    def speaker_verification(self,eval_path,eval_list,model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()
        for path in eval_path:
            for i, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
                spk = file.split('/')[0]
                audio, _  = soundfile.read(os.path.join(path, file))
                data_1 = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
                
               
                with torch.no_grad():
                    mask = self.model(data_1)
                    embedding, mel_pre = self.auxl(data_1, None, 'score',mask)
                    embedding_1 = F.normalize(embedding, p=2, dim=1)
                embeddings[file] = [embedding_1, ]
            scores, labels  = [], []
            for line in lines:
                embedding_11, = embeddings[line.split()[1]]
                embedding_21, = embeddings[line.split()[2]]
                score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
                score = score_1.detach().cpu().numpy()
                scores.append(score)
                labels.append(int(line.split()[0]))
            EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
            fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
            minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
            print('path:',path)
            print('EER:{}, minDCF:{}'.format(EER,minDCF))

    def apply(self):
        eval_path = '/data_a11/mayi/dataset/IRM/mix_v3'
        eval_list = '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb2/dev/sub_spk_list_v2'
        clean_path = '/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb2/dev/aac_split'
        self.labels = torch.load('exp/resnet_babble_speaker.pt')
        lines = open(eval_list).readlines()
        files = []
        accs=0
        for line in lines:
            lines = line.strip('\n').split('/')
            files.append(os.path.join(lines[-3],lines[-2],lines[-1]))
        
        setfiles = list(set(files))
        setfiles.sort()
        embeddings = {}
        for i, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):

            # Load data from validation_dataloader
            spk = file.split('/')[0]
            idx = int(spk[2:])
            num = str(idx//200)
            audio, _  = soundfile.read(os.path.join(eval_path, num,file))
            Xb = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
            audio, _  = soundfile.read(os.path.join(clean_path, num,file))
            clean = torch.FloatTensor(np.stack([audio],axis=0)).cuda()

            #feature = torch.load('/data_a11/mayi/project/SIP/IRM/exp/debug/feature.pt')
            with torch.no_grad():
                mask = self.model(Xb)
            target_spk = torch.stack([torch.tensor(int(self.labels[spk]))]).cuda()

            acc = self.auxl(Xb, target_spk, 'acc',mask)
            if acc.item()==1:
                continue
            print('wrong')
            SaM_c,mel_c,target,fea_c = self.layer_CAM(clean)
            SaM_pre,mel_pre,_,fea_pre = self.layer_CAM(Xb,target,mask)
            SaM_n,mel_n,_,fea_n = self.layer_CAM(Xb,target,None)

            weight = torch.sum(SaM_pre-SaM_c,dim=1).view(1,-1)
            weight = torch.nn.functional.softmax(weight, dim=1)
            weight = weight.view(1,1,SaM_c.shape[-2],SaM_c.shape[-1])
            max = torch.amax(weight,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
            min = torch.amin(weight,dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)
            weight=(weight-min)/(max-min)
            #continue
            

            if not os.path.exists(os.path.join(self.PROJECT_DIR,file.split('/')[-3],file.split('/')[-2])):
                os.makedirs(os.path.join(self.PROJECT_DIR,file.split('/')[-3],file.split('/')[-2]))
            #continue
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            #librosa.display.specshow(mel_n.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0,0], vmin=min,vmax=max)
            max = torch.max(mel_c)
            min = torch.min(mel_c)
            librosa.display.specshow(mel_c.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])

            librosa.display.specshow(mel_n.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1],vmin=min,vmax=max)
            
            img = librosa.display.specshow(mel_pre.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2],vmin=min,vmax=max)
            fig.colorbar(img, ax=ax)
            #librosa.display.specshow(pred_mel.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1,1], vmin=min,vmax=max)

            plt.savefig(os.path.join(self.PROJECT_DIR,file.replace('.wav','mask.png')))
            plt.close()
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            #librosa.display.specshow(mel_n.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0,0], vmin=min,vmax=max)
            max = torch.max(SaM_c)
            min = torch.min(SaM_c)
            librosa.display.specshow(torch.sum(SaM_c,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])

            librosa.display.specshow(torch.sum(SaM_pre,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1],vmin=min,vmax=max)
            img = librosa.display.specshow(weight.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
            fig.colorbar(img, ax=ax)
            #librosa.display.specshow(pred_mel.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1,1], vmin=min,vmax=max)

            plt.savefig(os.path.join(self.PROJECT_DIR,file.replace('.wav','sam.png')))
            plt.close()


            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            #librosa.display.specshow(mel_n.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0,0], vmin=min,vmax=max)
            max = torch.max(torch.sum(fea_c,dim=1))
            min = torch.min(torch.sum(fea_c,dim=1))
            librosa.display.specshow(torch.sum(fea_c,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])

            librosa.display.specshow(torch.sum(fea_pre,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1],vmin=min,vmax=max)
            
            img = librosa.display.specshow(torch.sum(torch.abs(fea_pre-fea_c),dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2],vmin=min,vmax=max)
            fig.colorbar(img, ax=ax)
            plt.savefig(os.path.join(self.PROJECT_DIR,file.replace('.wav','fea.png')))
            plt.close()

            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            #librosa.display.specshow(mel_n.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0,0], vmin=min,vmax=max)
            max = torch.max(torch.sum(fea_c,dim=1))
            min = torch.min(torch.sum(fea_c,dim=1))
            librosa.display.specshow(torch.sum(fea_c,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])

            librosa.display.specshow(torch.sum(fea_pre,dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1],vmin=min,vmax=max)
            img = librosa.display.specshow(torch.sum(weight*torch.abs(fea_pre-fea_c),dim=1).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2],vmin=min,vmax=max)
            print(weight.shape)
            fig.colorbar(img, ax=ax)
            plt.savefig(os.path.join(self.PROJECT_DIR,file.replace('.wav','weighted_fea.png')))
            plt.close()



        #print('accuracy',accs/len(setfiles))





