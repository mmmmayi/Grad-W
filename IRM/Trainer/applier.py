import os, torchaudio, soundfile,csv
from pystoi import stoi
from pypesq import pesq 
from tqdm import tqdm
import torch
import librosa
import librosa.display
import numpy as np
import tools.utils as utils
from Models.TDNN import ECAPA_TDNN, multi_TDNN
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
from Models.models import PreEmphasis
from tools.metrics import compute_PESQ, compute_segsnr
import torch.nn as nn
from Models.TDNN import Speaker_TDNN
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
        
        self.mean = torch.load("/data_a11/mayi/project/CAM/IRM/exp/mean.pt").float().cuda()
        self.std = torch.load("/data_a11/mayi/project/CAM/IRM/exp/std.pt").float().cuda()
        self.mean = self.mean.reshape(1,257,1)
        self.std = self.std.reshape(1,257,1)
        
        # Load model
        self.model = torch.load(model_path)
        self.loss = nn.MSELoss()
        self.model.eval()
        self.speaker = Speaker_TDNN().cuda()
        self_state = self.speaker.state_dict()
        path = '/data_a11/mayi/project/ECAPATDNN-analysis/exps/pretrain.model'
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            if name not in self_state:
                name = name.replace("speaker_encoder.","")
                if name not in self_state:
                    continue
            self_state[name].copy_(param)
        for p in self.speaker.parameters():
            p.requires_grad = False
        self.speaker.eval()


    def restore(self):
        torch.save(self.model.state_dict(),self.model_path.replace('.pt','_para.pt'))
        
    def __get_global_mean_variance(self):
        mean = 0.0
        variance = 0.0
        N_ = 0

        print("Calculating test dataset mean and variance...")
        for sample_batched in tqdm(self.applier_dataloader):
            Xb = sample_batched["mix_mag"].cuda()

            N_ += Xb.size(0)
            mean += Xb.mean(1).sum(0)
            variance += Xb.var(1).sum(0)

        mean = mean / N_
        variance = variance / N_
        standard_dev = torch.sqrt(variance)

        # Save global mean/var
        if not os.path.exists(f"{self.PROJECT_DIR}/{self.test_set_name}"):
            os.makedirs(f"{self.PROJECT_DIR}/{self.test_set_name}")
        torch.save(mean, f"{self.PROJECT_DIR}/{self.test_set_name}/global_mean.pt")
        torch.save(standard_dev, f"{self.PROJECT_DIR}/{self.test_set_name}/global_std.pt")

    def vari_sigmoid(self,x,a):
        return 1/(1+(-a*x).exp())

    def debug(self):
        loss_fn = nn.MSELoss()
        Xb = torch.load('/data_a11/mayi/project/CAM/IRM/exp/2D_msemean_BLSTM_cplxF_0.001_adadelta_logstft_20s/models/Xb.pt')
        Xb_input = ((Xb+1e-8).log()-self.mean)/self.std
        target_spk = torch.load('/data_a11/mayi/project/CAM/IRM/exp/2D_msemean_BLSTM_cplxF_0.001_adadelta_logstft_20s/models/spk.pt')        
        feature = Xb.requires_grad_()

        score = self.speaker(feature, target_spk.cuda(), 'score')
        self.speaker.zero_grad()
        yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
        cam = self.model(Xb, Xb_input, target_spk)

        for i in range(8):
            print(loss_fn(cam[i], yb[i]))
        
#Xb = Xb.reshape(1,Xb.size()[-1]).cuda()
        for i in range(8):
            #yb = self.vari_sigmoid(yb,20)
            
            name = str(i)
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            librosa.display.specshow(Xb[i].log().detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])
            img = librosa.display.specshow(yb[i].detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
            librosa.display.specshow(cam[i].log().detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
            fig.colorbar(img, ax=ax)
            plt.savefig(os.path.join('/data_a11/mayi/project/CAM/IRM/exp/2D_msemean_BLSTM_cplxF_0.001_adadelta_logstft_20s/debug',name+'.png'))
            plt.close()
       
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
        #num_0, num_5, num_10, num_15, num_20 = 0,0,0,0,0
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
        for sample_batched in tqdm(self.applier_dataloader):
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
            cam = self.model(mel_n, Xb_input)
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
        #print('SISNR:{}, SISNR_before:{}'.format(str(SISNR/num), str(SISNR_before/num))+'\n')
        #print('PESQ:{}, PESQ_before:{}'.format(str(PESQ/num), str(PESQ_before/num))+'\n')
        #print('STOI:{}, STOI_before:{}'.format(str(STOI/num), str(STOI_before/num))+'\n')
        #return 
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

    #@torch.no_grad()
    def apply(self):
        total_files = len(self.applier_dataloader)
        loss_fn = nn.MSELoss(reduction='sum')
        error_sum_origin, error_sum_target = 0,0
        #file = open(os.path.join(self.PROJECT_DIR,'x'),'w')
        #print(len(self.applier_dataloader))
        for sample_batched in tqdm(self.applier_dataloader):
            
            # Load data from validation_dataloader
            Xb, target_spk, clean, info = sample_batched
            target_spk = target_spk.reshape(1)
            assert Xb.size()[0] == 1, "The batch size of validation dataloader must be 1."
            if Xb.size()[1]==1:
                Xb = Xb.reshape(1,Xb.size()[-1]).cuda()
                clean = clean.reshape(1,Xb.size()[-1]).cuda()

            else:
                Xb = Xb.squeeze().cuda()
            
            Xb = self.pre(Xb)
            clean = self.pre(clean)
            #Xb = (self.MelSpec(Xb)+1e-6)
            Xb = (self.Spec(Xb)+1e-8).log()
            Xb_input = (Xb-self.mean)/self.std
            clean = (self.Spec(clean)+1e-8).log()
            feature = clean.requires_grad_()
            
            score = self.speaker(feature, target_spk.cuda(), 'score')
            self.speaker.zero_grad()
            yb = torch.autograd.grad(score, feature, grad_outputs=torch.ones_like(score), retain_graph=False)[0]
            #yb = self.vari_sigmoid(yb,1)
            B,H,W = yb.shape


            
            info = info[0].split('/')
            idx = int(info[-3][2:])
            num = str(idx//200)
            
            #truth = torch.load(os.path.join('/data_a11/mayi/dataset/noisy_gradient_input',num,info[0],info[1],info[2].replace('.wav','.pt')))
            #truth = torch.load(os.path.join('/mayi/mask/clean_gradient_input/',num,info[0],info[1],info[2].replace('.wav','.pt')))

            #cam_gt = (truth_norm+1e-6).log()+Xb
            #Xb_input = Xb_input.unsqueeze(1)
            cam = self.model(Xb, Xb_input, target_spk)
            
            #repre, target_repre, mask = self.model(Xb,cam_gt)
            #print('with origin:',self.loss(repre,origin))
            #print('with target:',self.loss(repre,target_repre))
            #file.write('with origin:'+str(self.loss(repre,origin).item())+'\n')
            #file.write('with target:'+str(self.loss(repre,target_repre).item())+'\n')
            #file.write('=========='+'\n')
            #error_sum_origin += self.loss(repre,origin).item()
            #error_sum_target += self.loss(repre,target_repre).item()
            #continue
            if not os.path.exists(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2])):
                os.makedirs(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2]))
            torch.save(cam, os.path.join(self.PROJECT_DIR,num,info[-3],info[-2],info[-1].replace('.wav','.pt'))) 
            continue
            m = nn.Sigmoid()
            #yb_ = yb.detach().cpu().squeeze()
            #yb_ = np.sort(yb_,axis=None).squeeze()
            #x = np.arange(len(yb_))
            #plt.scatter(x,yb_)
            #plt.savefig(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2],info[-1].replace('.wav','sort2.png')))
            #plt.close()
            
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            #librosa.display.specshow(Xb1.squeeze().detach().cpu().numpy(),x_axis=None, ax=ax[0])
            #librosa.display.specshow(Xb2.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
            librosa.display.specshow(Xb.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[0])

            #librosa.display.specshow(target_repre.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
            librosa.display.specshow(clean.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[1])
            #librosa.display.specshow(((cam+1e-8).log()+Xb).detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])
            img2 = librosa.display.specshow(cam.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[2])

            #librosa.display.specshow(irm.detach().cpu().squeeze().numpy(),x_axis=None, ax=ax[3])
            #fig.colorbar(img2, ax=ax)
            plt.savefig(os.path.join(self.PROJECT_DIR,num,info[-3],info[-2],info[-1].replace('.wav','.png')))
            plt.close()
            quit()
        #file.write('mse on origin'+str(error_sum_origin/len(self.applier_dataloader))+'\n')
        #file.write('mse on target'+str(error_sum_target/len(self.applier_dataloader))+'\n')
