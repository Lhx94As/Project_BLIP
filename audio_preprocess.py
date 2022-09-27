import os
import torch
import librosa
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf

# def resample(audio, tgt_rate):
#     new_name = audio.replace('.wav','_16k.wav')
#     subprocess.call(f"sox {audio} -r {tgt_rate} {new_name}", shell = True)
#     return new_name

def resample(root, target_sr=16000):
    data, sr = librosa.load(root, sr=None)
    if sr == target_sr:
        print('Do not need resampling for {}'.format(root))
    y_resample = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
    sf.write(root, y_resample, target_sr, subtype='PCM_16')

def audio_segmentation(audio, transcript):
    data, sr = librosa.load(audio, sr = None)
    if sr != 16000:
        print("resampling audio")
        resample(audio, target_sr=16000)
    root = os.path.split(audio)[0]
    audio_name = os.path.split(audio)[-1].split('.')[0]
    save_dir = root + f'/{audio_name}/'
    print(f"Audio chunks are saved in {save_dir}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(transcript, "r", encoding='utf-8') as f:
        lines = f.readlines()
    ind_ = 0
    time_stamps = {}
    labels = {}
    audio_list = {}
    data_ = AudioSegment.from_file(audio)
    for i in tqdm(range(len(lines))):
        line_ = lines[i]
        info_ = line_.split()
        if info_[1] == "(Language)" :
            if info_[-1].strip() == "English" \
                    or info_[-1].strip() == "Mandarin" \
                    or info_[-1].strip() == "reddot": # add more languages/sound types here
                audio_seg = audio_name + f'_{ind_}.wav'
                save_name = save_dir + audio_seg
                start = float(info_[2])
                end = float(info_[3])
                start_ = start * 1000
                end_ = end * 1000
                data_seg = data_[start_:end_]
                data_seg.export(save_name, format = 'wav') # save audio segments to a local folder /xxx/{audioname}/
                # save related dicts: audio_list, time stamps, and labels. Will be used for writing rttm
                audio_list[audio_seg] = save_name
                time_stamps[audio_seg] = f"{start} {end}"
                labels[audio_seg] = info_[-1]
                ind_ += 1
    return audio_list, time_stamps, labels

if __name__ == "__main__":
    audio_ = 'audio.wav'
    trans_ = 'trans.txt'
    audio_list, time_stamps, labels = audio_segmentation(audio_, trans_)
    # print(audio_list)






