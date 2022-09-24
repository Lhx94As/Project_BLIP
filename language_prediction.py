import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import X_Transformer_E2E_LID
from feat_extraction import *
from collections import OrderedDict


def language_prediction(pretrained_model, input_dim=1024, feature_list=None, labels=None, device="cpu"):
    model = X_Transformer_E2E_LID(input_dim=input_dim,
                                  feat_dim=64,
                                  d_k=64,
                                  d_v=64,
                                  d_ff=2048,
                                  n_heads=8,
                                  dropout=0.1,
                                  n_lang=2,
                                  max_seq_len=10000)

    model.to(device)
    pretrained_dict = torch.load(pretrained_model, map_location=device)
    new_state_dict = OrderedDict()
    model_dict = model.state_dict()
    dict_list = []
    for k, v in model_dict.items():
        dict_list.append(k)
    for k, v in pretrained_dict.items():
        if k.startswith('module.') and k[7:] in dict_list:
            new_state_dict[k[7:]] = v
        elif k in dict_list:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    valid_set = blip_dataset(feature_list, labels)
    valid_data = DataLoader(dataset=valid_set,
                            batch_size=1,
                            pin_memory=True,
                            shuffle=False,
                            collate_fn=collate_fn_atten)
    model.eval()
    correct = 0
    total = 0
    scores = {}
    predictions = {}
    score_matrix = 0
    audio_names = list(feature_list.keys())
    with torch.no_grad():
        for step, (utt, labels, seq_len) in enumerate(valid_data):
            audio_seg = audio_names[step]
            utt = utt.to(device=device, dtype=torch.float)
            labels = labels.to(device)
            # Forward pass\
            outputs = model(utt, seq_len)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(-1)
            correct += (predicted == labels).sum().item()
            score = F.softmax(outputs, dim = -1)
            score = score.squeeze().cpu().numpy()
            scores[audio_seg] = score
            predictions[audio_seg] = predicted
            if step == 0:
                score_matrix = outputs
            else:
                score_matrix = torch.cat((score_matrix, outputs), dim=0)
    acc = correct / total
    print('Current Acc.: {:.4f} %'.format(100 * acc))
    score_matrix = score_matrix.squeeze().cpu().numpy()
    return scores, predictions, acc



