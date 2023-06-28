from torch.utils.data import Dataset
import librosa, os, math, soundfile, random
import numpy as np
import torch
import torchaudio
import tools.utils as utils
from tools.Resampler import librosa_resample
import torch.nn.functional as F
from random import choice
from scipy import signal

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample["mix_mag"] = torch.from_numpy(sample["mix_mag"])
        sample["clean_mag"] = torch.from_numpy(sample["clean_mag"])
        sample["irm"] = torch.from_numpy(sample["irm"])
        sample["n_frames"] = torch.from_numpy(sample["n_frames"])
        return sample


class IRMDataset(Dataset):
    def __init__(self, path, spk,  batch_size, dur, path_sorted=None, sub=None, sampling_rate=16000, mode="train", center_num=4, test_num=2,num_spk=2):
        super().__init__()
        seed=0
        self.dur = dur
        self.center_num = int(center_num)
        self.test_num = int(test_num)
        self.num_spk = int(num_spk)
        random.seed(seed)
        np.random.seed(seed)
        if mode is not 'quality':
            spk = open(spk,'r')
            spk = [line.rstrip() for line in spk]
        self.mode = mode
        self.sampling_rate = sampling_rate
        if self.mode in ['train','validation']:
            path,self.spk_utt = sort(path, path_sorted, spk,dur, sub) ###todo, data contains paths

            self.data,self.spks = generate(path)
        index = -1
        self.batch_data = []
        number_of_minibathes = math.ceil(len(self.data) / batch_size)
        self.labels = torch.load('../lst/resnet_speaker.pt')
        for _ in range(number_of_minibathes):
            if index>=len(self.data)-batch_size:
                break
            self.batch_data.append([])
            for _ in range(batch_size):
                index = index+1
                self.batch_data[-1].append([self.data[index],self.spks[index]])
        with open ('/data_a11/mayi/dataset/musan/file_list', 'r') as f:
            self.noise_list=f.readlines()
        self.noise_num= len (self.noise_list)
        with open ('/data_a11/mayi/dataset/RIRS_NOISES/file_list', 'r') as f:
            self.rirs_list=f.readlines()
        self.rirs_num= len (self.rirs_list)


    def __len__(self):
        return len(self.batch_data)

    def shuffle(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.batch_data)))
        temp_data = [self.batch_data[i] for i in indices]
        self.batch_data = temp_data

    def __getitem__(self, idx):
        clean, noisy, noise, target = [], [], [],[]
        for pairs in self.batch_data[idx]:
            audio, _  = soundfile.read(pairs[0])
            spk = pairs[1]
            target_category = torch.tensor(int(self.labels[spk]))
        
            clip_input = audio.shape[0]
            start_point = np.random.randint(0, clip_input - self.dur*16000 + 1)
            end_point = start_point+self.dur*16000
            clean_audio = audio[start_point:end_point]
            noisy_audio,noise_audio = self.augment(audio[start_point:end_point],1)
            noise_tensor = torch.FloatTensor(np.stack([noise_audio],axis=0))
            clean_tensor = torch.FloatTensor(np.stack([clean_audio],axis=0))
            noisy_tensor = torch.FloatTensor(np.stack([noisy_audio],axis=0))
            clean.append(clean_tensor)
            noise.append(noise_tensor)
            noisy.append(noisy_tensor)
            target.append(target_category)
        return torch.stack(clean), torch.stack(noisy),  torch.stack(target), torch.stack(noise)
        '''
            spk = pairs[0].split('/')[-1].split('-')[0]
            target_category = torch.tensor(int(self.labels[spk]))
            
            if self.mode in ['train','validation']:
                #noise = self.generate_noise(audio)
                #snr = random.randint(0,20)
                #input_tensor = self.mix_waveform(audio, noise, snr)
                
                input_tensor = self.augment(audio)
                input_tensor = torch.FloatTensor(np.stack([input_tensor],axis=0))
                clean_tensor = torch.FloatTensor(np.stack([audio],axis=0)) 
            if len(noisy_input)==0:
                clip_input = input_tensor.shape[-1]
                start_point = 0
                end_point = clip_input
                if clip_input >15*16000:
                    start_point = np.random.randint(0, clip_input - 15*16000 + 1)
                    end_point = start_point+15*16000
            noisy_input.append(input_tensor[:,start_point:end_point])
            clean_input.append(clean_tensor[:,start_point:end_point])
            target_spk.append(target_category)
        if self.mode in ['train','validation']:
            return torch.stack(noisy_input), torch.stack(target_spk), torch.stack(clean_input), correct_spk
        '''
    def augment(self,audio, aug_prob=1):
        if aug_prob > random.random():
            aug_type = random.randint(1, 2)
            if aug_type == 0:
                #add reverb
                audio_len = audio.shape[0]
                num = np.random.randint(0, self.rirs_num, size=1)[0]
                rirs_path = self.rirs_list[num].strip('\n')
                rir_audio, _  = soundfile.read(rirs_path)
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            else:
                #add noise
                out_audio=audio

                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-4)
                num = np.random.randint(0, self.noise_num, size=1)[0]
                noise_path = self.noise_list[num].strip('\n')
                type = noise_path.split('/')[5]
                noise_list, snr_list = [],[]
                if 'noise' in type:
                    snr_list.append(random.uniform(-5, 10))
                    noise_list.append(self.generate_noise(audio,noise_path))
                elif 'speech' in type:
                    num_speech = np.random.randint(3,8)
                    snr_list.append(np.random.choice([5,7,10,13,15]))
                    noise_list.append(self.generate_noise(audio,noise_path))
                    while len(noise_list)<num_speech:
                        num = np.random.randint(0, self.noise_num, size=1)[0]
                        noise_path = self.noise_list[num].strip('\n')
                        type = noise_path.split('/')[5]
                        if 'speech' in type:
                            snr_list.append(np.random.choice([10,13,15,17,20]))
                            noise_list.append(self.generate_noise(audio,noise_path))

                elif 'music' in type:
                    snr_list.append(random.uniform(-5, 10))
                    noise_list.append(self.generate_noise(audio,noise_path))
                for i in range(len(noise_list)):
                    noise_snr = snr_list[i]
                    noise = noise_list[i]
                    noise_db = 10 * np.log10(np.mean(noise**2) + 1e-4)
                    noise_audio = np.sqrt(10**((audio_db - noise_db - noise_snr) / 10)) * noise
                    out_audio = out_audio + noise_audio

        else:
            out_audio = audio
        return out_audio, noise

    def generate_noise(self, audio,noise_path):
        
        noise, _  = soundfile.read(noise_path)
        len1 = len(audio)
        len2 = len(noise)

        if len2 <= len1:
        #if the noise's length  <speech's length, repeat it
            repeat_data= np.tile(noise, int(np.ceil(float(len1)/float(len2))))
            noise= repeat_data [0:len1]
        else:
        # randomly select data with equal length
            noise_onset=np.random.randint(low=0, high =len2- len1 , size=1)[0]
            noise_offset =noise_onset +len1
            noise = noise[noise_onset: noise_offset]
        return noise

    def mix_waveform(self,audio, noise, snr):

        audio = torch.FloatTensor(audio)
        noise = torch.FloatTensor(noise)
        audio_power = audio.norm(p=2)
        noise_power = noise.norm(p=2)
        snr = math.exp(snr / 10)
        scale = snr * noise_power / audio_power
        noisy_speech = (scale * audio + noise) / 2
        return noisy_speech.unsqueeze(0)

def sort(path, path_sorted, spk_list,dur, sub):
    f = open(path,'r')
    line = f.readline()
    utt_dict = {}
    while line:
        item = line.strip('\n')
        utt = item.split(' ')[0]
        n_signal = int(item.split(' ')[1])
        utt_dict[utt] = n_signal
        line = f.readline()
    total = len(utt_dict)
    num = int(sub*total)
    utts_temp,utts = [],[]
    i = 0
    spk_utt = {}
   
    if path_sorted is not None:
        f= open(path_sorted, 'r')

        for line in f.readlines()[8691:]:
            i = i+1
            item = line.strip('\n')
            utt = item.split(' ')[0]
            n_signal = utt_dict[utt]
            spk = utt.split('/')[0]
            if spk not in spk_list:
                continue
            if n_signal<dur*16000:
                continue
            utts_temp.append([utt,n_signal])
            if i>num:
                break
    else:
        spk_random = list(utt_dict.keys())        
        random.shuffle(spk_random)
        for spk_i in spk_random:
            spk = spk_i.split('/')[1].split('-')[0]
            if spk not in spk_list:
                continue
            if utt_dict[spk_i]<dur*16000:
                continue
            utts_temp.append([spk_i, utt_dict[spk_i]])
            if spk not in spk_utt:
                spk_utt[spk]=[spk_i]
            else:
                spk_utt[spk].append(spk_i)
            i+=1
            if i>num:
                break
    sort_utts = sorted(utts_temp,key=lambda info:int(info[1]),reverse=False)
    for utt in sort_utts:
        utts.append(utt[0])
    print('the shortest one:{}, the longest one:{}'.format(sort_utts[0][1],sort_utts[-1][1]))
    return utts, spk_utt

def generate(path):
    data,label = [],[]
    for i in path:
        idx = i.split('/')[0]
        spk = i.split('/')[1].split('-')[0]
        utt = i.split('/')[1]
        
        data.append(os.path.join('/data_a11/mayi/dataset/voxceleb_asvtorch/VoxCeleb2/dev/acc_new/',idx,spk,utt))
        label.append(spk)
    return  data,label

def generate_test(path, mode, spk_list=None):
    f = open(path,'r')
    line = f.readline()
    data, name = [], []
    while line:
        item = line.strip('\n')
        utt = item.split(' ')[0]
        if mode=='quality':
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr0',utt))
            name.append(utt)
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr5',utt))
            name.append(utt)
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr10',utt))
            name.append(utt)
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr15',utt))
            name.append(utt)
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb1/test/snr20',utt))


            #data.append(os.path.join('/data_a11/mayi/dataset/DNS-test/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy',utt))
            line = f.readline()
            name.append(utt)
            continue

        spk = utt.split('/')[0]
        idx = int(spk[2:])
        num = str(idx//200)
        if spk not in spk_list:
            line = f.readline()
            continue
        if mode in ['clean_single', 'clean_multi']:
            data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb2/dev/aac_split/', num, utt))

        elif mode in ['noisy_single', 'noisy_multi', 'validation']:
            data.append(os.path.join('/data_a11/mayi/dataset/IRM/mix', num, utt))
        else:
            print('bug in dataloader')
            quit()
        line = f.readline()
        name.append(utt)
    return data, name
