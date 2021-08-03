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


class AudioPreprocessor(AudioExplorer):
    def __init__(self, directory:str, audio_dir:str=r'/wav/*.wav', tts_file:str=r'/trsTrain.txt'):
        try:
            super().__init__(directory,audio_dir,tts_file)
            e_logger.info('Successfully Inherited AudioExplorer Class')
        except Exception as e:
            e_logger.exception("Failed Inheriting")

        # so now we want to load the audio and transcription
        # but first we need to change the load
        self.load()

    
    


if __name__ == "__main__":
    ap = AudioPreprocessor(directory='../data/train')
    print(ap.get_tts().keys())