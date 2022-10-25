import os
import librosa
from librosa.core.spectrum import power_to_db
import numpy as np
from scipy import signal
import scipy.io.wavfile as wav_io
import librosa

def stft(x, window, nperseg=400, noverlap=240):
    if len(window)!=nperseg:
        raise ValueError('window length must equal nperseg')
    x=np.array(x)
    nadd = noverlap - (len (x) -nperseg)%noverlap
    x =np.concatenate((x, np.zeros(nadd)))
    step = nperseg - noverlap
    shape=x.shape[:-1] + ((x.shape[-1] - noverlap) //step, nperseg)
    strides=x.strides[:-1] + (step *x.strides[-1], x.strides[-1])
    x=np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x =x* window
    result= np.fft.rfft(x, n=nperseg)
    return result

def read_wav(path):
    sampling_rate, x = wav_io.read(path)
    return sampling_rate, x

def save_wav(audio_np, sr, path):
    wav_io.write(path, sr, np.int16(audio_np))

def wav2lps(x, window=np.hamming(400), nperseg=400, noverlap=240):
    y =stft(x, window, nperseg=nperseg, noverlap=noverlap)
    return np.log(np.square(abs(y))+1e-8)

def lps2mfcc(lps,n_fft=400,sr=16000):
    _spectrogrsm = np.e**lps
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=40)
    melspectrogram = np.dot(mel_basis, _spectrogrsm.T)
    S = power_to_db(melspectrogram)
    mfcc = librosa.feature.mfcc(S=S)
    return mfcc


def lps2wav(lps, wave, window=np.hamming(400), nperseg=400, noverlap=240):
    z = stft(wave, window, nperseg=nperseg, noverlap=noverlap)
    angle=z/ (np.abs(z) + 1e-8 )
    x=np.sqrt(np.exp(lps))* angle
    x=np.fft.irfft (x)
    y=np.zeros((len(x) -1) * (nperseg-noverlap) + nperseg)
    C1= window[0: nperseg-noverlap]
    C2= window[0: noverlap]  + window[nperseg-noverlap: nperseg]
    C3= window[noverlap: nperseg]
    y[0: nperseg-noverlap] =x[0][0: nperseg-noverlap] / C1
    for i in range (1, len(x)):
        y[i*(nperseg-noverlap):(i-1) *(nperseg-noverlap)+nperseg] = (x[i-1][nperseg-noverlap:nperseg] + x[i][0:noverlap])/C2
    y[-(nperseg-noverlap):] =x[len(x) -1][noverlap:] /C3
    return np.int16(y[0: len(wave)])

def lps2irm(lps_noise, lps_clean, window=np.hamming(400), nperseg=400, noverlap=240):
    return np.sqrt( np.exp(lps_clean)/(np.exp(lps_clean) + np.exp(lps_noise) ))
