#!/usr/bin/env python3
import librosa
import argparse, os
import scipy.io.wavfile as wav_io
import soundfile as sf


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="src_folder",
        type=str,
        help="Input folder")
    parser.add_argument(
        dest="dest_folder",
        type=str,
        help="Output folder")
    parser.add_argument("--resampling_rate", dest="resampling_rate", type=int, default=16000)
    return parser.parse_args()


def get_all_wav(input_folder:str) -> list:
    file_list = []
    for r, d, f in os.walk(input_folder):
        for file in f:
            if file.endswith(".wav"):
                file_list.append(os.path.join(r, file))
    return file_list


def resample(file:str, output_folder:str, resampling_rate:int):
    # Get filename
    filename = file.split('/')[-1]

    # convert
    y, sr = librosa.load(file, sr=None)
    y_ = librosa.resample(y, sr, resampling_rate, scale=True)

    # Write file
    sf.write("{}/{}".format(output_folder, filename), y_, resampling_rate, subtype='PCM_16')


if __name__ == "__main__":
    args = parse_arguments()

    # Create output folder
    if not os.path.exists(args.dest_folder): os.makedirs(args.dest_folder)

    # Convert all wav in input folder
    wav_list = get_all_wav(args.src_folder)
    for wav in wav_list:
        resample(
            file=wav, output_folder=args.dest_folder,
            resampling_rate=args.resampling_rate)
