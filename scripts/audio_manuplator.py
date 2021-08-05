from librosa.core import audio
from librosa.filters import mel
import numpy as np
import random
import pandas as pd
import numpy as np
import librosa
import soundfile
import os

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger
e_logger = CreateLogger('AudioPreprocessor')
e_logger = e_logger.get_default_logger()


class AudioManipulator:
    def __init__(self, df_audio,standard_sr=22500) -> None:
        try:
            self.df_audio = df_audio
            self.standard_sr = standard_sr
            # self.features_df = None
            self.max_dur = None
            self.get_max_duration()
            e_logger.exception("successfully Initialized AudioPreprocessor class!")
        except Exception as e:
            e_logger.exception("Failed Initializing")

            
 

    def audio_preprocess_extract(self):
        pass

    def get_max_duration(self):
        self.max_dur= float(round(self.df_audio["Duration"].max(),3))

    
    def convert_stereo_audio(self):
        converted_data = (self.df_audio).apply(lambda row: to_stereo(row["TimeSeriesData"]),axis=1)
        update_channel = self.df_audio["TimeSeriesData"].apply(lambda row: check_channels(row))
        self.df_audio["TimeSeriesData"] = converted_data
        self.df_audio["Channels"] = update_channel
        
        
    def resize_audio(self):
        converted_data = self.df_audio.apply(lambda row:pad_silence(row["TimeSeriesData"],row["SamplingRate"],self.max_dur),axis=1)
        self.df_audio["TimeSeriesData"] = converted_data
        update_duration = self.df_audio["TimeSeriesData"].apply(lambda row: round(librosa.get_duration(row),3))
        self.df_audio["Duration"] = update_duration
    
    def resample_audio(self):
        converted_data = self.df_audio.apply(lambda row:change_sr(row["TimeSeriesData"],row["SamplingRate"],self.standard_sr),axis=1)
        self.df_audio["TimeSeriesData"] = converted_data
        self.df_audio["SamplingRate"] = self.standard_sr
    
    def write_wave_files(self,path):
        self.df_audio.apply(lambda row: write_wav(path+row["Name"]+".wav",np.transpose(row["TimeSeriesData"]),row["SamplingRate"]),axis=1)
        
    def extract_features(self):
        pass


    def get_audio_info(self):
        return self.df_audio

def write_wav(filename,y,sr):
    soundfile.write(filename, y, sr)

def to_stereo(y):
# if it isnot stereo
    if (y.shape[0] != 2):
        y = np.vstack((y, y))
    return y

def check_channels(y):
        channel =""
        if y.shape[0] == 1:
            channel ="Mono"
        else:
            channel ="Stereo"
        return channel

def pad_silence(y, sr, max_s):
    num_rows, y_len = y.shape
    max_len = sr * max_s
    print(max_len)
#     Truncate the signal to the given length
#     if (y_len > max_len):
#         y = y[:,:max_len]
    
    if(y_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - y_len)
        pad_end_len = int(max_len - y_len - pad_begin_len)
        # Pad with 0s
        pad_begin = np.zeros((num_rows, pad_begin_len))
#         print(type(num_rows),type(pad_end_len))
        pad_end = np.zeros((num_rows, pad_end_len))
        y = np.hstack((pad_begin, y, pad_end))      
    return y

def change_sr(y,sr, new_sr):
    y = librosa.resample(y,sr,new_sr)
    return y
    
def extract_melspectograms(y,sr,hop_len=512, win_len=1024, n_mels=128):
# extract melspectogram from the audio file using librosa library
    melspect = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_len, win_length=win_len, n_mels=n_mels,fmax=sr/2)
    melspect_db = librosa.power_to_db(melspect, ref=np.max)
    return melspect,melspect_db
