import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import logging
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils
import s3prl.upstream.wav2vec2.hubconf as hubconf
from sklearn.preprocessing import LabelEncoder
from audio_preprocess import *



def feat_segmentation(feature, overlap=5, chunk_len=20):
    new_feat = 0
    seq_len = feature.shape[0]
    step = chunk_len - overlap
    num_chunk = (seq_len - overlap) // (chunk_len - overlap)
    if num_chunk > 1:
        start = 0
        end = 0
        for id in range(num_chunk):
            end = start + chunk_len
            feat_temp = feature[start:end, :]
            feat_temp = np.hstack(feat_temp)
            start += step
            if id == 0:
                new_feat = feat_temp
            else:
                new_feat = np.vstack((new_feat, feat_temp))
        num_left = seq_len - end
        start = end - (chunk_len - num_left)
        feat_temp = feature[start:, :]
        feat_temp = np.hstack(feat_temp)
        new_feat = np.vstack((new_feat, feat_temp))
    return new_feat

def mfcc_feat_extraction(audio_list, order_1=True, order_2=True, mfccdim=13):
    """
    :param audio_list: a python dictionary {audio_seg: seg_path}
    :param order_1: path of pre-trained XLSR-53 model, default model/xlsr_53_56k.pt
    :param order_2: from which layer you'd like to extract the wav2vec 2.0 features, default 14
    :param mfccdim: dimension of mfcc dim, default 13 (39 after being stacked with its 1st and 2nd orders)
    :return: a python dictionary {audio_seg: features}
    """
    audio_names = list(audio_list.keys())
    feature_output = {}
    for i in tqdm(range(len(audio_names))):
        audio = audio_list[audio_names[i]]
        audioarray, sr_ = librosa.load(path=audio, sr=None)
        preemphasis = 0.97
        preemphasized = np.append(audioarray[0], audioarray[1:] - preemphasis * audioarray[:-1])
        mfcc = librosa.feature.mfcc(y = preemphasized, sr = sr_, n_mfcc=mfccdim,
                                    hop_length=int(sr_ / 100), n_fft=int(sr_ / 40))
        if order_1 and order_2:
            delta1 = librosa.feature.delta(mfcc, order=1)
            delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_features = np.vstack((mfcc, delta1, delta2))
        else:
            mfcc_features = mfcc
        mfcc_features_seg = feat_segmentation(mfcc_features.T)
        if not isinstance(mfcc_features_seg, int):
            feature_output[audio_names[i]] = mfcc_features_seg
    return feature_output

def w2v_feat_extraction(audio_list, model_path="models/xlsr_53_56k.pt", layer=14, device="cpu"):
    """
    :param audio_list: a python dictionary {audio_seg: seg_path}
    :param model_path: path of pre-trained XLSR-53 model, default model/xlsr_53_56k.pt
    :param layer: from which layer you'd like to extract the wav2vec 2.0 features, default 14
    :param device: cpu or gpu, default cpu in case that gpu does not exist, but recommend gpu
    :return: feature_output: a python dictionary {audio_seg: features}
    """
    feat_layer = layer
    feature_output = {}
    audio_names = list(audio_list.keys())
    model = hubconf.wav2vec2_local(ckpt=model_path)
    model.to(device)
    for i in tqdm(range(len(audio_names))):
        audio = audio_list[audio_names[i]]
        data, sr = librosa.load(audio, sr=None)
        try:
            data_ = torch.tensor(data).to(device=device, dtype=torch.float).unsqueeze(0)
            features = model(data_)
            features = features['hidden_state_{}'.format(feat_layer)]
            features_ = features.squeeze(0).detach().cpu().numpy()
            features_seg = feat_segmentation(features_)
            if not isinstance(features_seg, int):
                feature_output[audio_names[i]] = features_seg
        except:
            print(f"{audio} is too short")
    return feature_output

def collate_fn_atten(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seq, labels, seq_length = zip(*batch)
    data = rnn_utils.pad_sequence(seq, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return data, labels, seq_length

class blip_dataset(data.Dataset):
    def __init__(self, feature_output, labels):
        self.audio_names = list(feature_output.keys())
        self.feature_list = [feature_output[i] for i in self.audio_names]
        self.label_list = [labels[i] for i in self.audio_names]
        le = LabelEncoder()
        le.fit(["English", "Mandarin"])
        self.label_list = le.transform(self.label_list)
        self.seq_len_list = [i.shape[0] for i in self.feature_list]

    def __getitem__(self, index):
        feature = torch.from_numpy(self.feature_list[index])
        label = int(self.label_list[index])
        seq_len = int(self.seq_len_list[index])
        return feature, label, seq_len

    def __len__(self):
        return len(self.label_list)




