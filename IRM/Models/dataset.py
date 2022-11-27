from torch.utils.data import Dataset
import librosa, os, math, soundfile, random
import numpy as np
import torch
import torchaudio
import tools.utils as utils
from tools.Resampler import librosa_resample
import torch.nn.functional as F

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample["mix_mag"] = torch.from_numpy(sample["mix_mag"])
        sample["clean_mag"] = torch.from_numpy(sample["clean_mag"])
        sample["irm"] = torch.from_numpy(sample["irm"])
        sample["n_frames"] = torch.from_numpy(sample["n_frames"])
        return sample


class IRMDataset(Dataset):
    def __init__(self, path, spk,  batch_size, dur, path_sorted=None, sub=None, sampling_rate=16000, mode="train", max_size=100000, data='both'):
        super().__init__()
        if mode is not 'quality':
            spk = open(spk,'r')
            spk = [line.rstrip() for line in spk]
        self.mode = mode
        self.sampling_rate = sampling_rate
        if self.mode in ['train','validation']:
            path = sort(path, path_sorted, spk,dur, sub) ###todo, data contains paths
            self.noisy_data, self.noisy_label = generate(path, self.mode)
        else:
            self.noisy_data, self.noisy_label= generate_test(path, mode, spk)
        index = -1
        self.batch_data = []
        self.labels = torch.load('../lst/resnet_speaker.pt')
        number_of_minibathes = math.ceil(len(self.noisy_data) / batch_size)
        for _ in range(number_of_minibathes):
            if index>=len(self.noisy_data)-batch_size:
                break
            self.batch_data.append([])
            for _ in range(batch_size):   
                index = index+1
                self.batch_data[-1].append([self.noisy_data[index],self.noisy_label[index]])
        with open ('/data_a11/mayi/dataset/musan/file_list', 'r') as f:
            self.noise_list=f.readlines()
        self.noise_num= len (self.noise_list)

    def __len__(self):
        return len(self.batch_data)

    def shuffle(self, seed):
        np.random.seed(seed)
        indices = np.random.permutation(np.arange(len(self.batch_data)))
        temp_data = [self.batch_data[i] for i in indices]
        self.batch_data = temp_data

    def __getitem__(self, idx):
        clean_input, noise_input = [],[]
        noisy_input, target_spk = [],[]
        check = []
        
        for pairs in self.batch_data[idx]:
            check.append(pairs[0])
            audio, _  = soundfile.read(pairs[0])
            if self.mode == 'quality':
                input_tensor = torch.FloatTensor(np.stack([audio],axis=0))
                #clean_path = pairs[0].replace('/noisy/','/clean/')
                #wav_name = pairs[0].split('/')[-1]
                #subs = wav_name.split('_')
                #new_name = 'clean_'+subs[-2]+'_'+subs[-1]
                #clean_path = clean_path.replace(wav_name,new_name)  
                clean_path = pairs[0].replace(pairs[0].split('/')[-4],'wav')
                clean_audio, _  = soundfile.read(clean_path)
                clean_tensor = torch.FloatTensor(np.stack([clean_audio],axis=0))
                return input_tensor, clean_tensor

            spk = pairs[1].split('/')[-3]
            target_category = torch.tensor(int(self.labels[spk]))

            if self.mode=='train':
                noise = self.generate_noise(audio)
                snr = random.randint(0,20)
                input_tensor = self.mix_waveform(audio, noise, snr)
                clean_tensor = torch.FloatTensor(np.stack([audio],axis=0))
            else: 
                input_tensor = torch.FloatTensor(np.stack([audio],axis=0))
                clean_path = pairs[0].replace('IRM/mix','VoxCeleb_latest/VoxCeleb2/dev/aac_split')
                clean_audio, _  = soundfile.read(clean_path)
                clean_tensor = torch.FloatTensor(np.stack([clean_audio],axis=0))

            if len(noisy_input)==0:
                clip_input = input_tensor.shape[-1]
            noisy_input.append(input_tensor[:,0:clip_input])
            clean_input.append(clean_tensor[:,0:clip_input])
            target_spk.append(target_category)
        if self.mode in ['train','validation']:
            return torch.stack(noisy_input), torch.stack(target_spk), torch.stack(clean_input), check

    def generate_noise(self, audio):
        num = np.random.randint(0, self.noise_num, size=1)[0]
        noise_path = self.noise_list[num].strip('\n')
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
    f= open(path_sorted, 'r')
    utts_temp,utts = [],[]
    i = 0
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
    print(len(utts_temp))
    sort_utts = sorted(utts_temp,key=lambda info:int(info[1]),reverse=False)
    for utt in sort_utts:
        utts.append(utt[0])
    print('the shortest one:{}, the longest one:{}'.format(sort_utts[0][1],sort_utts[-1][1]))
    return utts

def generate(path,mode):
    clean_data,clean_label = [],[]
    noisy_data,noisy_label = [],[]
    for i in path:
        spk = i.split('/')[0]
        idx = int(spk[2:])
        num = str(idx//200)
        
        if mode=='train':
            noisy_data.append(os.path.join('/data_a11/mayi/dataset/VoxCeleb_latest/VoxCeleb2/dev/aac_split',num,i))
        else:
            noisy_data.append(os.path.join('/data_a11/mayi/dataset/IRM/mix',num,i))
        noisy_label.append(os.path.join('/data_a11/mayi/dataset/noisy_gradient_input/',num,i))
  
    return  noisy_data, noisy_label

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
