import librosa

def librosa_resample(x, sampling_rate:int, new_resampling_rate:int):
    x_ = librosa.resample(x, sampling_rate, new_resampling_rate, scale=True)
    return x_