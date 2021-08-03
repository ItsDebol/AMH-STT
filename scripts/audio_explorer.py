# imports
import pandas as pd
import numpy as np
import librosa as lb
from glob import glob
from json import dump

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger

logger = CreateLogger('AudioExplorer', handlers=1)
logger = logger.get_default_logger()

class AudioExplorer:
    def __init__(self, directory:str, audio_dir:str=r'/wav/*.wav', tts_file:str=r'/trsTrain.txt') -> None:
        try:
            self.tts_dict = {}
            self.df = None
            self.main_dir = directory
            self.audio_files= glob(self.main_dir + audio_dir)
            self.tts_file = self.main_dir + tts_file
            logger.info('Successfully Created AudioExplorer Class')
            self.load()
            logger.info('Successfully Loaded Audio and TTS files')

        except Exception as e:
            logger.exception("Failed Creating the AudioExplorer Class")

    def load(self):
        self.load_tts()
        self.load_audio()
    
    def load_tts(self) -> None:
        try:
            with open(self.tts_file, encoding='UTF-8') as tts_handle:
                lines = tts_handle.readlines()
                for line in lines:
                    transliteration, file_name = line.split('</s>')
                    transliteration = transliteration.replace('<s>', '').strip()
                    file_name = file_name.strip()[1:-1]
                    self.tts_dict[file_name] = transliteration
        except FileNotFoundError as e:
            logger.exception(f'File {self.tts_file} doesnt exist in the directory')
        except Exception as e:
            logger.exception('Failed to Load Transliteration File')


    def get_tts(self) -> dict:
        try:
            return self.tts_dict
        except Exception as e:
            logger.exception('Failed to return Transliteration')

    def export_tts(self, file_name:str) -> None:
        try:
            with open(file_name, "w") as export_file:
                dump(self.tts_dict, export_file, indent=4, sort_keys=True)

            logger.info(f'Successfully Exported Transliteration as JSON file to {file_name}.')

        except FileExistsError as e:
            logger.exception(f'Failed to create {file_name}, it already exists.')
        except Exception as e:
            logger.exception('Failed to Export Transliteration as JSON File.')

    def check_tts_exist(self, file_name:str) -> bool:
        if(file_name in self.tts_dict.keys()):
            return True
        else:
            return False

    def get_tts_value(self, file_name:str) -> str:
        if(self.check_tts_exist(file_name)):
            return self.tts_file[file_name]
        else:
            return 'Unknown'

    def load_audio(self) -> None:

        try:
            audio_name = []
            audio_amplitude_min = []
            audio_amplitude_max = []
            audio_amplitude_mean = []
            audio_amplitude_median = []
            audio_frequency = []
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
                # Minimum Audio Amplitude
                audio_amplitude_min.append(round(min(audio_data),3))
                # Maximum Audio Amplitude
                audio_amplitude_max.append(round(max(audio_data),3))
                # Mean Audio Amplitude
                audio_amplitude_mean.append(round(np.mean(audio_data), 3))
                # Median Audio Amplitude
                audio_amplitude_median.append(round(np.median(audio_data), 3))
                # Audio Frequency
                audio_frequency.append(audio_freq)
                # TTS
                tts_status = self.check_tts_exist(name)
                has_TTS.append(tts_status)
                # Add Transliteration
                if(tts_status):
                    tts.append(self.tts_dict[name])
                else:
                    tts.append(None)
                
            self.df = pd.DataFrame()
            self.df['Name'] = audio_name
            self.df['Duration'] = audio_duration
            self.df['Frequency'] = audio_frequency
            self.df['MinAmplitude'] = audio_amplitude_min
            self.df['MaxAmplitude'] = audio_amplitude_max
            self.df['AmplitudeMean'] = audio_amplitude_mean
            self.df['AmplitudeMedian'] = audio_amplitude_median
            self.df['HasTTS'] = has_TTS
            self.df['TTS'] = tts

        except Exception as e:
            logger.exception('Failed to Load Audio Files')

    def get_audio_info(self) -> pd.DataFrame:
        try:
            return self.df.drop('TTS',axis=1)
        except Exception as e:
            logger.exception('Failed to return Audio Information')
    
    def get_audio_info_with_tts(self) -> pd.DataFrame:
        try:
            return self.df
        except Exception as e:
            logger.exception('Failed to return Audio Information')

    def get_audio_files(self) -> list:
        try:
            return self.audio_files
        except Exception as e:
            logger.exception('Failed to return Audio Files')

    def get_audio_file(self, index:int):
        try:
            return self.audio_files[index]
        except IndexError as e:
            logger.exception(f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to return Audio File')

    def get_audio_file_info(self, index:int):
        try:
            return self.df.iloc[index,:]
        except  IndexError as e:
            logger.exception(f"Audio Files only exist between 0 - {len(self.df) - 1}")
        except Exception as e:
            logger.exception('Failed to return Audio File')

    


if __name__ == "__main__":
    ae = AudioExplorer(directory='../data/train')
    print(ae.get_tts())
    print(ae.get_audio_info())
