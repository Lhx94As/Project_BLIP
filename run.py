import glob
import os.path
from tqdm import tqdm
from language_prediction import *
from feat_extraction import *
from audio_preprocess import *
from prepare_rttm import *


def main():
    parser = argparse.ArgumentParser(description='paras for making data')
    parser.add_argument('--audio', type=str,
                        help="Path of the input BLIP recording")
    parser.add_argument('--trans', type=str,
                        help="Path of transcript corresponding to the input BLIP recording")
    parser.add_argument('--feat_type', type=str, default="wav2vec",
                        help="choose features type, either mfcc or wav2vec features")
    parser.add_argument('--lid_model', type=str,
                        help="models/lid.ckpt Pre-trained LID model")
    parser.add_argument('--w2v_model', type=str, default=None,
                        help="models/xlsr_53_56k.pt, don't need to input if using mfcc")
    parser.add_argument('--w2v_layer', type=int, default=14)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Starting audio segmentation via the transcript...")
    audio_path = args.audio
    trans_path = args.trans
    if audio_path.endswith('.wav'):
        audio_list, time_stamps, labels = audio_segmentation(audio_path, trans_path)
        feature_output = None
        input_dim = None
        print("Starting {} feature extraction".format(args.feat_type))
        if args.feat_type == "mfcc":
            feature_output = mfcc_feat_extraction(audio_list)
            input_dim = 39
        elif args.feat_type == "wav2vec":
            feature_output = w2v_feat_extraction(audio_list, args.w2v_model, args.w2v_layer, device)
            input_dim = 1024
        print("Starting language prediction...")
        scores, predictions = language_prediction(args.lid_model, input_dim, feature_output, labels, device)
        print("starting writing rttm")
        write_rttm(predictions, scores, labels, time_stamps, args.trans)
    else:
        audio_path_list = glob.glob(audio_path+"/*wav")
        trans_path_list = glob.glob(trans_path+"/*txt")
        trans_dict = {}
        for trans_ in trans_path_list:
            trans_name = os.path.split(trans_)[-1]
            trans_key = f"{trans_name.splt('_')[0]}_{trans_name.splt('_')[1]}_" \
                        f"{trans_name.splt('_')[2]}_{trans_name.splt('_')[3]}_{trans_name.splt('_')[4]}"
            trans_dict[trans_key] = trans_name

        for _ind in tqdm(range(len(audio_path_list))):
            audio_ = audio_path_list[_ind]
            audio_name = os.path.split(audio_)[-1]
            audio_key = f"{audio_name.splt('_')[0]}_{audio_name.splt('_')[1]}_" \
                        f"{audio_name.splt('_')[2]}_{audio_name.splt('_')[3]}_{audio_name.splt('_')[4]}"
            trans_ = trans_dict[audio_key]
            audio_list, time_stamps, labels = audio_segmentation(audio_, trans_)
            feature_output = None
            input_dim = None
            if args.feat_type == "mfcc":
                feature_output = mfcc_feat_extraction(audio_list)
                input_dim = 39
            elif args.feat_type == "wav2vec":
                feature_output = w2v_feat_extraction(audio_list, args.w2v_model, args.w2v_layer, device)
                input_dim = 1024
            scores, predictions = language_prediction(args.lid_model, input_dim, feature_output, labels, device)
            write_rttm(predictions, scores, labels, time_stamps, args.trans)





if __name__ == "__main__":
    main()



