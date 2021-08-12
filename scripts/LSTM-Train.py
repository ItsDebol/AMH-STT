
import tensorflow as tf
import scipy.io.wavfile as wav
import glob
import numpy as np
import matplotlib.pyplot as plt

import json
from python_speech_features import mfcc, logfbank
import mlflow
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Activation, Bidirectional, TimeDistributed, Masking, Input, GRU, SimpleRNN
from tensorflow.keras.models import Model


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FEAT_MASK_VALUE = 1e+10

# Some configs
num_features = 13
num_classes = 222 + 1  # 285(including space) + blank label = 286

# Hyper-parameters
num_epochs = 300
batch_size = 100
initial_learning_rate = 0.0005
momentum = 0.9


# Loading the data
file_path = glob.glob('../data/train/clean_wav/*.wav')

audio_list = []
fs_list = []
min_duration = 2
max_duration = 6
new_file_path = []

for file_name in file_path:
    fs, audio = wav.read(file_name)
    audio_size = audio.size
    duration = audio_size / fs
    if(duration >= min_duration and duration <= max_duration):
        new_file_path.append(file_name)
        audio_list.append(audio)
        fs_list.append(fs)


# Create a dataset composed of data with variable lengths
inputs_list = []
input_type = 'mfcc'
for index in range(len(audio_list)):
    if(input_type == 'mfcc'):
        input_val = mfcc(audio_list[index],
                         samplerate=fs_list[index], numcep=13)
        input_val = (input_val - np.mean(input_val)) / np.std(input_val)
        inputs_list.append(input_val)
    else:
        num_features = 161
        input_val = logfbank(
            audio_list[index], samplerate=fs_list[index], nfilt=161)
        input_val = (input_val - np.mean(input_val)) / np.std(input_val)
        inputs_list.append(input_val)


# Transform in 3D Array
train_inputs = tf.ragged.constant([i for i in inputs_list], dtype=np.float32)
train_seq_len = tf.cast(train_inputs.row_lengths(), tf.int32)
train_inputs = train_inputs.to_tensor(default_value=FEAT_MASK_VALUE)


with open('../data/train_labels.json', 'r', encoding='UTF-8') as label_file:
    labels = json.load(label_file)
with open('../data/alphabets_data.json', 'r', encoding='UTF-8') as alphabets_file:
    alphabets = json.load(alphabets_file)

# update number of labels
num_classes = alphabets['alphabet_size'] + 1


# Reading Targets
original_list = []
targets_list = []

for path in new_file_path:
    file_name = path[:-4].split('wav')[1][1:].split('#')[0]
    # Read Label
    label = labels[file_name]
    original = " ".join(label.strip().split(' '))
    original_list.append(original)
    # print(original)
    target = original.replace(' ', '  ')
    # print('step-1. ',target)
    target = target.split(' ')
    # print('step-2. ', target)
    # Adding blank label
    target = np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target])
    # print('step-3. ', target)
    # Transform char into index
    target = np.asarray([alphabets['char_to_num'][x] for x in target])
    # print('step-4. ', target)
    targets_list.append(target)


# Creating sparse representation to feed the placeholder
train_targets = tf.ragged.constant([i for i in targets_list], dtype=np.int32)
train_targets_len = tf.cast(train_targets.row_lengths(), tf.int32)
train_targets = train_targets.to_sparse()


size, row = train_targets.shape
train_size = int(size * 0.8)
val_size = size - train_size
print(train_size, val_size)
print(row)


# Split Training and Validation sets

train_inputs_final, val_inputs_final = train_inputs[:
                                                    train_size], train_inputs[train_size:]
train_seq_len_final, val_seq_len_final = train_seq_len[:
                                                       train_size], train_seq_len[train_size:]
train_targets_final, val_targets_final = tf.sparse.slice(train_targets, start=[0, 0], size=[
                                                         train_size, row]), tf.sparse.slice(train_targets, start=[train_size, 0], size=[train_size, row])
train_targets_len_final, val_targets_len_final = train_targets_len[
    :train_size], train_targets_len[train_size:]


print(train_inputs_final.shape, val_inputs_final.shape)
print(train_seq_len_final.shape, val_seq_len_final.shape)
print(train_targets_final.shape, val_targets_final.shape)
print(train_targets_len_final.shape, val_targets_len_final.shape)


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


# Definning Input Parameters
input_feature = tf.keras.layers.Input(
    (None, num_features), name='input_feature')
input_label = tf.keras.layers.Input(
    (None,), dtype=tf.int32, sparse=True, name='input_label')
input_feature_len = tf.keras.layers.Input(
    (1,), dtype=tf.int32, name='input_feature_len')
input_label_len = tf.keras.layers.Input(
    (1,), dtype=tf.int32, name='input_label_len')


input_masking = tf.keras.layers.Masking(FEAT_MASK_VALUE)(input_feature)
x = tf.keras.layers.LSTM(100, return_sequences=True)(input_masking)
x_1 = tf.keras.layers.BatchNormalization()(x)
x_2 = tf.keras.layers.LSTM(100, return_sequences=True)(x_1)
x_3 = tf.keras.layers.BatchNormalization()(x_2)
x_4 = tf.keras.layers.LSTM(100, return_sequences=True)(x_3)
x_5 = tf.keras.layers.BatchNormalization()(x_4)
x_6 = tf.keras.layers.LSTM(100, return_sequences=True)(x_5)
x_7 = tf.keras.layers.BatchNormalization()(x_6)
x_8 = tf.keras.layers.LSTM(100, return_sequences=True)(x_7)
# x = tf.keras.layers.BatchNormalization()(x)
# layer_rnn = tf.keras.layers.LSTM(10, return_sequences=True)(layer_bn)
# x = tf.keras.layers.Dropout(0.2, seed=42)(x)
layer_output = tf.keras.layers.TimeDistributed(Dense(num_classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(
    0.0, 0.1), bias_initializer='zeros', name='logit'))(x_8)

layer_loss = CTCLossLayer()(
    [input_label, layer_output, input_label_len, input_feature_len])


# Create models for training and prediction
model_train = tf.keras.models.Model(inputs=[input_feature, input_label, input_feature_len, input_label_len],
                                    outputs=layer_loss)
print(model_train.summary())
model_predict = tf.keras.models.Model(
    inputs=input_feature, outputs=layer_output)


# Compile Training Model with selected optimizer
optimizer = tf.keras.optimizers.SGD(initial_learning_rate, momentum)
model_train.compile(optimizer=optimizer)

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='../models/'+"RNN"+'.h5',
                                                  monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# ModelCheckpoint(filepath='../models/'+"RNN"+'.h5', verbose=0,)

mlflow.set_experiment("STACKED LSTM Layers")
mlflow.tensorflow.autolog()
history = model_train.fit(x=[train_inputs_final, train_targets_final, train_seq_len_final, train_targets_len_final], y=None,
                          validation_data=(
                              [val_inputs_final, val_targets_final, val_seq_len_final, val_targets_len_final], None),
                          batch_size=batch_size, epochs=num_epochs)


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.show()
