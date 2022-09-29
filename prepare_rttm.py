import os
import numpy as np

def write_rttm(predictions, scores, labels, time_stamps, speaker, acc, orifile=None):
    """
    :param prediction: {audio_seg: 0 or 1}, 0 for English, 1 for Mandarin
    :param score: {audio_seg: score}, used as confidence
    :param labels: {audio_seg: labels}, ground truth labels can be either English or Mandarin
    :param time_stamps: {audio_seg: time stamps}
    :param speaker: {audio seg: speaker_id}, speaker id of the audio chunk
    :param acc: {acc}, language prediction accuracy for the entire audio file
    :param orifile: text file path
    :return:
    """

    for k, v in predictions.items(): #transform digits to labels in text
        if v == 0:
            predictions[k] = "English"
        else:
            predictions[k] = "Mandarin"

    audio_segs = list(scores.keys())
    if orifile is not None:
        out_file = orifile.replace(".txt","_output_rttm.txt")
    else:
        out_file = "{}_output_rttm.txt".format(audio_segs[0])

    with open(out_file, 'w') as f:
        f.write("Audio_name start end prediction confidence label\n")
        for i, seg_name in enumerate(audio_segs):
            f.write("{} {} {} {} {} {}\n".format(seg_name,   # replace space with tab
                                            time_stamps[seg_name],
                                            speaker[seg_name],
                                            predictions[seg_name],
                                            np.max(scores[seg_name], axis=-1),
                                            labels[seg_name]))
        f.write('LID Accuracy is: {:.4f} %'.format(100 * acc))
    print(f"Completed. The rttm is saved in {out_file}")
