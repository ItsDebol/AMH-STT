import sys
import os
import librosa as lb
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))
from audio_explorer import AudioExplorer

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
        self.load()

    def load_audio(self):

        try:
            audio_name = []
            audio_sr = []
            audio_duration = []
            audio_data = []
            has_TTS = []
            tts = []

            for audio_file in self.audio_files:
                data, audio_freq = lb.load(audio_file)
                # Audio_Name
                name = audio_file.split('wav')[-2]
                name = name[1:-1].strip()
                audio_name.append(name)
                # Time in seconds
                audio_duration.append(round(lb.get_duration(audio_data),3))
                # Audio Sampling Rate
                audio_sr.append(audio_freq)
                # Audio time series data
                audio_data.append(data)
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
            self.df['SamplingRate'] = audio_sr
            self.df['TimeSeriesData'] = audio_data
            self.df['HasTTS'] = has_TTS
            self.df['TTS'] = tts

        except Exception as e:
            e_logger.exception('Failed to Load Audio Files')


    


if __name__ == "__main__":
    ap = AudioPreprocessor(directory='../data/train')
    print(ap.get_tts().keys())