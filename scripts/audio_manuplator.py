from librosa.core import audio
from librosa.filters import mel
import numpy as np
import random
import pandas as pd
import numpy as np
import librosa


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
            self.features_df = None
            e_logger.exception("successfully Initialized AudioPreprocessor class!")
        except Exception as e:
            e_logger.exception("Failed Initializing")


    def audio_preprocess_extract(self):
        self.convert_to_mono()
        self.resample(self.standard_sr)
        self.extract_features()

    def convert_to_mono(self):
        stereo_mask = self.df_audio.query("Channels == 'Stereo'")
        self.df_audio.loc[stereo_mask,"TimeSeriesData"] = self.df_audio.loc[stereo_mask,"TimeSeriesData"].apply(lambda x: librosa.to_mono(x))
        return self.df_audio

    def resample(self,new_audio_sr):
        unsimilar_sr_mask = self.df["SamplingRate"] != new_audio_sr
        self.df.loc[unsimilar_sr_mask,"SamplingRate"] = self.df_audio.loc[unsimilar_sr_mask,:].apply(lambda x: librosa.resample(x["TimeSeriesData"],x["SamplingRate"],new_audio_sr))
        return self.df_audio

    def resize(self):
        pass

    def convert_to_db(S,ref):
        melspect_db = librosa.power_to_db(S, ref=ref)
        return melspect_db

    def extract_features(self,hop_len=512, win_len=1024, n_mels=128):
    # extract melspectogram from the audio file using librosa library
        melspectogram = []
        melspectogram_db = []

        for y,sr in zip(self.df_audio["TimeSeriesData"],self.df_audio["SamplingRate"]):
            melspect = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_len, win_length=win_len, n_mels=n_mels,fmax=sr/2)
            melspectogram.append(melspect)
            melspect_db = self.convert_to_db(melspect, ref=np.max)
            melspectogram_db.append(melspect_db)
        self.features_df = pd.DataFrame()
        self.features_df['Name'] = self.df_audio["Name"]
        self.features_df['Melspectogram'] = melspectogram
        self.features_df['Melspectogram_db'] = melspectogram_db
        return self.features_df


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

    def get_audio_features(self):
        self.audio_preprocess_extract()
        return self.features_df

