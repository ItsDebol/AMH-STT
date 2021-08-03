import numpy as np
import random
import pandas as pd
import numpy as np
import librosa

class Audio_Manuplator():
    def __init__(self,audio,sr) -> None:
        self.audio = audio
        self.sr = sr
        """
        Initialize the Class assigning its object 
        a new audio file and sampling rate 
        Parameters
        ----------
        audio - Audio file
        sr - Sampling rate

        Returns
        -------
        None
        """
    def convert_to_channels(self):
        pass

    def resample(self,new_sr):
        # resample audio
        return librosa.resample(self.audio,self.sr,new_sr)
    
    def resize(self):
        pass

    def extract_features(self,hop_len=512, win_len=1024, n_mels=128):
        # extract melspectogram from the audio file using librosa library
        melspect = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, hop_len=hop_len, win_len=win_len, n_mels=n_mels)
        melspect_db = librosa.power_to_db(melspect, ref=np.max)
        return melspect,melspect_db


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