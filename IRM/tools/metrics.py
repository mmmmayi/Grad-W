import numpy as np
import soundfile as sf
from pypesq import pesq
from pystoi.stoi import stoi


def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(clean_signal, noisy_signal, sr)

def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)

def compute_segsnr(clean_signal, noisy_signal, sr=16000, frame:int=0, overlap:int=0):
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
