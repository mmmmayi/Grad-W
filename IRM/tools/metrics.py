import numpy as np
import soundfile as sf
from pypesq import pesq
from pystoi.stoi import stoi
from operator import itemgetter
def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    return tunedThreshold, eer, fpr, fnr

def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm
    fnrs = [x / float(fnrs_norm) for x in fnrs]
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

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
