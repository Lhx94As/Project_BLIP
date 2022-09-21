import subprocess


print("Install torch")
subprocess.call("conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch", shell = True)

print("Install librosa")
subprocess.call("conda install -c conda-forge librosa", shell = True)

print("Install soundfile")
subprocess.call("conda install -c conda-forge pysoundfile", shell = True)

print("Install sklearn")
subprocess.call("conda install -c anaconda scikit-learn", shell = True)

print("Install tqdm")
subprocess.call("conda install -c conda-forge tqdm", shell = True)

print("Install pydub")
subprocess.call("conda install -c conda-forge pydub", shell = True)

print("Install sox")
subprocess.call("sudo apt-get -y install sox", shell = True)

print("Install s3prl")
subprocess.call("pip install s3prl", shell = True)

print("Install argparse")
subprocess.call("conda install -c conda-forge argparse", shell = True)

print("Install collection")
subprocess.call("conda install -c lightsource2-tag collection", shell = True)
