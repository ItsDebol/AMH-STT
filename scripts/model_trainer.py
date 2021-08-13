from logging import log
import tensorflow as tf
import scipy.io.wavfile as wav
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from json import load
from python_speech_features import mfcc, logfbank
import mlflow
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Activation, Bidirectional, TimeDistributed, Masking, Input, Dropout, GRU, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

try:
    from logger_creator import CreateLogger
except:
    from scripts.logger_creator import CreateLogger

# Initializing Logger
logger = CreateLogger('ModelTrainer', handlers=1)
logger = logger.get_default_logger()


class CTCLossLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        labels = inputs[0]
        logits = inputs[1]
        label_len = inputs[2]
        logit_len = inputs[3]

        logits_trans = tf.transpose(logits, (1, 0, 2))
        label_len = tf.reshape(label_len, (-1,))
        logit_len = tf.reshape(logit_len, (-1,))
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels, logits_trans, label_len, logit_len, blank_index=-1))
        # define loss here instead of in compile
        self.add_loss(loss)

        # Decode
        decoded, _ = tf.nn.ctc_greedy_decoder(logits_trans, logit_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), labels))
        self.add_metric(ler, name='ler', aggregation='mean')

        return logits


class ModelTrainer():
    # Constants
    SPACE_TOKEN = '<space>'
    FEAT_MASK_VALUE = 1e+10
    SAMPLE_RATE = 16000
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9

    def __init__(self, audios_path: str = '../data/train/clean_wav/', labels_path: str = '../data/train_labels.json', alphabes_path: str = '../data/alphabets_data.json', feature_used: str = 'mfcc', epoch: int = 300, learning_rate: float = 0.0005, duration_range: Tuple[int, int] = (2, 6)) -> None:
        try:
            self.audios_path = audios_path
            self.labels_path = labels_path
            self.alphabets_path = alphabes_path
            self.feature_used = feature_used
            self.num_epoch = epoch
            self.duration_range = duration_range
            ModelTrainer.LEARNING_RATE = learning_rate

            logger.info(
                'Successfully Created Model Trainer Class Object Instance')

        except Exception as e:
            logger.exception(
                'Failed to Create ModelTrainer Class Object Instance')

    def change_constants(self, mask_value=FEAT_MASK_VALUE, sample_rate=SAMPLE_RATE, learning_rate=LEARNING_RATE, momentum=MOMENTUM):
        try:
            ModelTrainer.FEAT_MASK_VALUE = mask_value
            ModelTrainer.SAMPLE_RATE = sample_rate
            ModelTrainer.LEARNING_RATE = learning_rate
            ModelTrainer.MOMENTUM = momentum

            logger.info('Successfully Changed ModelTrainer Constants')

        except Exception as e:
            logger.exception(
                'Failed to Change ModelTrainer Class Constants')

    def load_audio_files(self):
        # Loading the data
        try:
            self.files_path = glob.glob(self.audios_path + '*.wav')

            self.audio_list = []
            min_duration = self.duration_range[0]
            max_duration = self.duration_range[1]
            self.train_audio_paths = []

            for file_name in self.files_path:
                fs, audio = wav.read(file_name)
                audio_size = audio.size
                duration = audio_size / fs
                if(duration >= min_duration and duration <= max_duration):
                    self.train_audio_paths.append(file_name)
                    self.audio_list.append(audio)

            logger.info('Successfully Loaded Audio Files')

        except Exception as e:
            logger.exception('Failed To Load Audio Files')

    def extract_feature(self):
        # Create a dataset composed of data with variable lengths
        try:
            self.inputs_list = []
            for index in range(len(self.audio_list)):
                if(self.feature_used == 'mfcc'):
                    self.num_features = 13
                    input_val = mfcc(self.audio_list[index],
                                     samplerate=ModelTrainer.SAMPLE_RATE, numcep=13)
                    input_val = (input_val - np.mean(input_val)) / \
                        np.std(input_val)
                    self.inputs_list.append(input_val)
                elif(self.feature_used == 'logmelfb'):
                    self.num_features = 161
                    input_val = logfbank(
                        self.audio_list[index], samplerate=ModelTrainer.SAMPLE_RATE, nfilt=161)
                    input_val = (input_val - np.mean(input_val)) / \
                        np.std(input_val)
                    self.inputs_list.append(input_val)
                else:
                    raise('Unsupported/Invalid Feature Provided')

            logger.info(
                f'Successfully Extracted {self.feature_used} features from Audio Files')

        except Exception as e:
            logger.exception(
                f'Failed to Extract {self.feature_used} features from Audio Files')

    def construct_inputs(self):
        try:
            # Get Features
            self.extract_feature()
            # Transform in 3D Array
            self.train_inputs = tf.ragged.constant(
                [i for i in self.inputs_list], dtype=np.float32)
            self.train_seq_len = tf.cast(
                self.train_inputs.row_lengths(), tf.int32)
            self.train_inputs = self.train_inputs.to_tensor(
                default_value=ModelTrainer.FEAT_MASK_VALUE)

            logger.info('Successfully Constructed Model Inputs')

        except Exception as e:
            logger.exception('Failed to Construct Model Inputs')

    def load_labels_and_alphabets(self):
        try:
            with open(self.labels_path, 'r', encoding='UTF-8') as label_file:
                self.labels = load(label_file)

            with open(self.alphabets_path, 'r', encoding='UTF-8') as alphabets_file:
                self.alphabets = load(alphabets_file)

            # Extract Number of Classes(Alphabet CHaracters) + blank label(1)
            self.num_classes = self.alphabets['alphabet_size'] + 1

            logger.info('Successfully Load Labels and Alphabets')

        except Exception as e:
            logger.exception('Failed to Load Labels and Alphabets')

    def extract_transciptions(self):
        try:
            # Load Labels and Alphabets
            self.load_labels_and_alphabets()
            # Reading Targets
            self.original_list = []
            self.targets_list = []

            for path in self.train_audio_paths:
                file_name = path[:-4].split('wav')[1][1:].split('#')[0]
                # Read Label
                label = self.labels[file_name]
                original = " ".join(label.strip().split(' '))
                self.original_list.append(original)

                target = original.replace(' ', '  ')
                target = target.split(' ')
                # Adding blank label
                target = np.hstack(
                    [ModelTrainer.SPACE_TOKEN if x == '' else list(x) for x in target])
                # Transform char into index
                target = np.asarray([self.alphabets['char_to_num'][x]
                                     for x in target])
                self.targets_list.append(target)

            logger.info('Successfully Extracted Transcriptions')

        except Exception as e:
            logger.exception('Failed to Extract Transcriptions')

    def construct_targets(self):
        try:
            # Extract Transcriptions
            self.extract_transciptions()
            # Creating sparse representation to feed the placeholder
            self.train_targets = tf.ragged.constant(
                [i for i in self.targets_list], dtype=np.int32)
            self.train_targets_len = tf.cast(
                self.train_targets.row_lengths(), tf.int32)
            self.train_targets = self.train_targets.to_sparse()

            logger.info('Successfully Constructed Model Targets')

        except Exception as e:
            logger.exception('Failed to Construct Model Targets')

    def split_train_validation_sets(self):
        try:
            # Split to 80%,20%
            size, row = self.train_targets.shape
            train_size = int(size * 0.8)
            # val_size = size - train_size
            # Split Training and Validation sets
            self.train_inputs_final, self.val_inputs_final = self.train_inputs[
                :train_size], self.train_inputs[train_size:]
            self.train_seq_len_final, self.val_seq_len_final = self.train_seq_len[
                :train_size], self.train_seq_len[train_size:]
            self.train_targets_final, self.val_targets_final = tf.sparse.slice(self.train_targets, start=[0, 0], size=[
                train_size, row]), tf.sparse.slice(self.train_targets, start=[train_size, 0], size=[train_size, row])
            self.train_targets_len_final, self.val_targets_len_final = self.train_targets_len[
                :train_size], self.train_targets_len[train_size:]

            logger.info('Successfully Split Train and Validation Sets')

        except Exception as e:
            logger.exception('Failed to Split Train and Validation Sets')

    def get_input_target_shapes(self):
        print('INPUT SHAPES: \n\t', 'TRAIN: ', self.train_inputs_final.shape,
              '\n\tVALIDATION: ', self.val_inputs_final.shape)
        print('INPUT SEQUENCE LENGTH SHAPES: \n\t', 'TRAIN: ', self.train_seq_len_final.shape,
              '\n\tVALIDATION: ',  self.val_seq_len_final.shape)
        print('TARGET SHAPES: \n\t', 'TRAIN: ', self.train_targets_final.shape,
              '\n\tVALIDATION: ',  self.val_targets_final.shape)
        print('TARGET SEQUENCE LENGTH SHAPES: \n\t', 'TRAIN: ', self.train_targets_len_final.shape,
              '\n\tVALIDATION: ',  self.val_targets_len_final.shape)

    def define_model_input_params(self):
        try:
            # Definning Input Parameters
            self.input_feature = tf.keras.layers.Input(
                (None, self.num_features), name='input_feature')
            self.input_label = tf.keras.layers.Input(
                (None,), dtype=tf.int32, sparse=True, name='input_label')
            self.input_feature_len = tf.keras.layers.Input(
                (1,), dtype=tf.int32, name='input_feature_len')
            self.input_label_len = tf.keras.layers.Input(
                (1,), dtype=tf.int32, name='input_label_len')

            logger.info('Successfully Defined Model Input Parameters')

        except Exception as e:
            logger.exception('Failed to Define Model Input Parameters')

    def get_stacked_LSTM_model(self):
        try:
            input_masking = Masking(
                ModelTrainer.FEAT_MASK_VALUE)(self.input_feature)

            x = LSTM(100, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                                     dropout=0.2, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True)(input_masking)
            x_1 = BatchNormalization()(x)
            x_2 = LSTM(100, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                                       dropout=0.1, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True)(x_1)
            x_3 = BatchNormalization()(x_2)
            x_4 = LSTM(100, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                                       dropout=0, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True)(x_3)
            x_5 = BatchNormalization()(x_4)
            x_6 = LSTM(50, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                       dropout=0.1, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True)(x_5)
            x_7 = BatchNormalization()(x_6)
            x_8 = LSTM(50, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                       dropout=0, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True)(x_7)
            self.layer_output = TimeDistributed(Dense(self.num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(
                0.0, 0.1), bias_initializer='zeros', name='logit'))(x_8)

            self.layer_loss = CTCLossLayer()(
                [self.input_label, self.layer_output, self.input_label_len, self.input_feature_len])

            logger.info('Successfully Defined STACKED LSTM MODEL')

        except Exception as e:
            logger.exception('Failed to define Stacked LSTM Model')

    def get_bidirectional_lstm_model(self):
        try:
            input_masking = Masking(
                ModelTrainer.FEAT_MASK_VALUE)(self.input_feature)

            x = Bidirectional(LSTM(256, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                                                   dropout=0.2, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True),
                              backward_layer=LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                                                  kernel_regularizer=l2(0.01), dropout=0.2, recurrent_dropout=0, use_bias=True,
                                                  unroll=False, return_sequences=True))(input_masking)

            x_1 = BatchNormalization()(x)
            x_2 = Bidirectional(LSTM(256, activation='tanh', recurrent_activation='sigmoid', kernel_regularizer=l2(0.01),
                                     dropout=0.2, recurrent_dropout=0, use_bias=True, unroll=False, return_sequences=True),
                                backward_layer=LSTM(256, activation='tanh', recurrent_activation='sigmoid',
                                                    kernel_regularizer=l2(0.01), dropout=0.2, recurrent_dropout=0, use_bias=True,
                                                    unroll=False, return_sequences=True))(input_masking)
            x_3 = BatchNormalization()(x_2)
            x_4 = TimeDistributed(
                Dense(self.num_classes, activation='maxout', kernel_regularizer=l2(0.01)))(x_3)
            x_5 = Dropout(0.2)(x_4)
            x_6 = TimeDistributed(
                Dense(self.num_classes, activation='maxout', kernel_regularizer=l2(0.01)))(x_5)
            x_7 = Dropout(0.1)(x_6)

            self.layer_output = TimeDistributed(Dense(self.num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(
                0.0, 0.1), bias_initializer='zeros', name='logit'))(x_7)

            self.layer_loss = CTCLossLayer()(
                [self.input_label, self.layer_output, self.input_label_len, self.input_feature_len])

            logger.info('Successfully Defined Bidirectional LSTM MODEL')

        except Exception as e:
            logger.exception('Failed to define Bidirectional LSTM Model')

    def create_train_predict_models(self, model_name: str = 'stacked_lstm'):
        try:
            models = {'stacked_lstm': self.get_stacked_LSTM_model,
                      'bidirectional_lstm': self.get_bidirectional_lstm_model}
            # Create models for training and prediction
            if(model_name in models.keys()):
                models[model_name]()
                self.model_train = Model(inputs=[self.input_feature, self.input_label, self.input_feature_len, self.input_label_len],
                                         outputs=self.layer_loss)

                self.model_predict = Model(
                    inputs=self.input_feature, outputs=self.layer_output)

                print(self.model_train.summary())

            else:
                raise('Unsupported/Invalid Model Given')

            logger.info('Successfully Created Train and Predict Models')

        except Exception as e:
            logger.exception('Failed to Create Train and Predict Models')

    def train_model(self, model_name: str, mlflow_experiment: str, batch_size: int, optimizer: str = 'adam', learning_rate: float = LEARNING_RATE, momentum: float = MOMENTUM, checkpoint_save: bool = True, early_stop: bool = False):
        try:
            # Compile Training Model with selected optimizer
            optimizers = {'sgd': tf.keras.optimizers.SGD(
                learning_rate, momentum), 'adam': tf.keras.optimizers.Adam(learning_rate)}
            if(optimizer in optimizers.keys()):
                optimizer = optimizers[optimizer]

                # Compile Model
                self.model_train.compile(optimizer=optimizer)

                checkpointer_callback = tf.keras.callbacks.ModelCheckpoint(filepath='../models/'+model_name+'_train.h5',
                                                                           monitor='val_ler', verbose=1, save_best_only=True, mode='min')

                earlystop_callback = tf.keras.callbacks.EarlyStopping(
                    monitor="val_ler", min_delta=0.001, patience=2)

                callbacks = []
                if checkpoint_save == True:
                    callbacks.append(checkpointer_callback)
                if early_stop == True:
                    callbacks.append(earlystop_callback)

                mlflow.set_experiment(mlflow_experiment)
                mlflow.tensorflow.autolog()
                self.history = self.model_train.fit(x=[self.train_inputs_final, self.train_targets_final, self.train_seq_len_final, self.train_targets_len_final], y=None,
                                                    validation_data=(
                                                        [self.val_inputs_final, self.val_targets_final, self.val_seq_len_final, self.val_targets_len_final], None),
                                                    batch_size=batch_size, epochs=self.num_epoch, callbacks=callbacks)

                # Save Predicton Model
                self.model_predict.save(
                    '../models/' + model_name + '_predict.h5')

            else:
                raise('Unsupported/Invalid Optimizer Given')

            logger.info('Successfully Trained Model')

        except Exception as e:
            logger.exception('Failed to Train Model')

    def draw_result_plots(self, model_name: str, loss_type: int, title: str = '', size: Tuple[int, int] = (10, 5)):
        try:
            # print(history.history.keys())
            # summarize history for accuracy
            if(loss_type == 1):
                plt.figure(figsize=size)
                plt.plot(self.history.history['loss'])
                plt.plot(self.history.history['val_loss'])
                plt.title(
                    'MODEL LOSS DIAGRAM') if title == "" else plt.title(title)
                plt.ylabel('LOSS')
                plt.xlabel('EPOCH')
                plt.legend(['Training Loss', 'Validation Loss'],
                           loc='upper right')
                plt.savefig('../models/' + model_name + '-loss.jpg')
                plt.show()

            elif(loss_type == 2):
                plt.figure(figsize=size)
                plt.plot(self.history.history['ler'])
                plt.plot(self.history.history['val_ler'])
                plt.title(
                    'MODEL LER ERROR DIAGRAM') if title == "" else plt.title(title)
                plt.ylabel('ERROR RATE')
                plt.xlabel('EPOCH')
                plt.legend(['Training LER', 'Validation LER'],
                           loc='upper right')
                plt.savefig('../models/' + model_name + '-ler.jpg')
                plt.show()

            else:
                raise('Unsupported/Invalid LOSS TYPE Given (only 1 and 2 available)')

            logger.info(f'Successfully Ploted Loss Type: {loss_type}')

        except Exception as e:
            logger.exception('Failed To Plot History Result')

    def get_sample_trained_model_transcriptions(self, amount: int = 2, check_augmentation: bool = False):
        try:
            # Decoding
            size = self.val_inputs_final.shape
            size = size[0]
            amount *= 7
            if amount > size:
                amount = size

            print('Original:')
            for index, org_label in enumerate(self.original_list):
                if(check_augmentation):
                    if(index + 1 > amount):
                        print(
                            'Effects are order: #BG1, #BG2, #BG3, #CLEAN, #GAUSIAN_NOISE, #VOL_INCREASE, #VOL_DECREASE')
                        break
                    print(index, '. ', org_label)
                else:
                    if(index + 1 > amount):
                        break
                    elif(index % 7 == 0):
                        print(index, '. ', org_label)
            print('Decoded:')
            decoded, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.model_predict.predict(
                self.val_inputs_final[:amount]), (1, 0, 2)), self.val_seq_len_final[:amount])
            d = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
            str_decoded = [''.join([self.alphabets['num_to_char'][str(x)]
                                    for x in np.asarray(row) if x != -1]) for row in d]
            for index, prediction in enumerate(str_decoded):
                # Replacing blank label to none
                # s = s.replace(chr(ord('z') + 1), '')
                if(check_augmentation):
                    prediction = prediction.replace(
                        self.alphabets['num_to_char']['0'], ' ')
                    print(index, '. ', prediction)
                else:
                    if(index % 7 == 0):
                        # Replacing space label to space
                        prediction = prediction.replace(
                            self.alphabets['num_to_char']['0'], ' ')
                        print(index, '. ', prediction)

            logger.info(
                'Successfully Produced Predicted Transcription Samples')

        except Exception as e:
            logger.exception(
                'Failed to Get Transcription Samples From Trained Model')


if __name__ == '__main__':

    model_trainer = ModelTrainer(learning_rate=0.001)
    model_trainer.load_audio_files()

    model_trainer.construct_inputs()
    model_trainer.construct_targets()

    model_trainer.split_train_validation_sets()
    model_trainer.get_input_target_shapes()

    model_trainer.define_model_input_params()
    model_trainer.create_train_predict_models(model_name='stacked_lstm')
    model_trainer.train_model(
        model_name='stacked-lstm', mlflow_experiment='STACKED LSTM Layers', batch_size=500, optimizer='adam')

    model_trainer.draw_result_plots(1)
    model_trainer.draw_result_plots(2)

    model_trainer.get_sample_trained_model_transcriptions()
