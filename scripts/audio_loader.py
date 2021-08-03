import sys
import os
import librosa as lb
from librosa.core import audio
import pandas as pd
try:
    from audio_explorer import AudioExplorer
except:
    from scripts.audio_explorer import AudioExplorer

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger

e_logger = CreateLogger('AudioPreprocessor')
e_logger = e_logger.get_default_logger()


class AudioLoader(AudioExplorer):
    def __init__(self, directory:str, audio_dir:str=r'/wav/*.wav', tts_file:str=r'/trsTrain.txt'):
        try:
            super().__init__(directory,audio_dir,tts_file)
            e_logger.info('Successfully Inherited AudioExplorer Class')
        except Exception as e:
            e_logger.exception("Failed Inheriting")

        # so now we want to load the audio and transcription
        # but first we need to change the load

    def load_audio(self):

        try:
            audio_name = []
            audio_frequency = []
            audio_ts_data = []
            audio_duration = []
            has_TTS = []
            tts = []

            for audio_file in self.audio_files:
                audio_data, audio_freq = lb.load(audio_file)
                # Audio_Name
                name = audio_file.split('wav')[-2]
                name = name[1:-1].strip()
                audio_name.append(name)
                # Time in seconds
                audio_duration.append(round(lb.get_duration(audio_data),3))
                # Audio Sampling Rate
                audio_frequency.append(audio_freq)
                # Audio time series data
                audio_ts_data.append(audio_data)
                # TTS
                tts_status = self.check_tts_exist(name)
                has_TTS.append(tts_status)
                # Add Transcription
                if(tts_status):
                    tts.append(self.tts_dict[name])
                else:
                    tts.append(None)
                
            self.df = pd.DataFrame()
            self.df['Name'] = audio_name
            self.df['Duration'] = audio_duration
            self.df['SamplingRate'] = audio_frequency
            self.df['TimeSeriesData'] = audio_ts_data
            self.df['HasTTS'] = has_TTS
            self.df['TTS'] = tts
            print(self.df.loc[0,"Name"])
        except Exception as e:
            e_logger.exception('Failed to Load Audio Files')

    def get_audio_info(self) -> pd.DataFrame:
        try:
            return self.df.drop(columns=['TTS','TimeSeriesData'],axis=1)
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

    def get_audio_info_with_data_tts(self) -> pd.DataFrame:
        try:
            self.df
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

    def get_audio_info_with_data(self) -> pd.DataFrame:
        try:
            return self.df.drop('TTS',axis=1)
        except Exception as e:
            e_logger.exception('Failed to return Audio Information')

# if __name__ == "__main__":
#     al = AudioLoader(directory='data/train')
#     print(al.get_audio_info().head(3))