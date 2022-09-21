# Project_BLIP_recipe

Before performing LID, 
>sudo python download_dependency_linux/windows.py  

according to your operation system.  

It is recommended to also download `fairseq` and `tensorxboard` using pip install. Note that for Python 3.2 and above, `argparse` and `collection` are included. 

To perform LID,  
>python run.py --audio your_audio_path --trans corresponding_transcript_path --feat_type wav2vec --lid_model your_lid_model_path --w2v_model pretrained_wav2vec_model_path --w2v_layer 14  
