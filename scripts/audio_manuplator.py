import numpy as np
import random
import pandas as pd
import numpy as np
import librosa

def convert_to_mono(y):
    return librosa.to_mono(y)

def resample(y,sr,new_sr):
    # resample audio
    return librosa.resample(y,sr,new_sr)

def resize(self):
    pass

def extract_features(y,sr,hop_len=512, win_len=1024, n_mels=128):
    # extract melspectogram from the audio file using librosa library
    melspect = librosa.feature.melspectrogram(y=y, sr=sr, hop_len=hop_len, win_length=win_len, n_mels=n_mels)
    return melspect

def convert_to_db(melspect):
    melspect_db = librosa.power_to_db(melspect, ref=np.max)
    return melspect_db


def melspect_resize(melspect,n_mels=128):
    # takes a melspectogram and pads it, if it is more than
    # 128 and truncates if it is less 
    melspect_len = melspect.shape[1]

    if melspect_len < n_mels:
        melspect = librosa.util.fix_length(melspect, size=n_mels, axis=1,constant_values=(0, -80.0))
    if melspect_len > n_mels:
        random_split = random.randint(0,melspect_len-n_mels)
        melspect = melspect[:,random_split:random_split+n_mels]
    return melspect

      

