
# Importing Class
import sys
import os
from model_trainer import ModelTrainer

# Creating Model Trainer Class
model_trainer = ModelTrainer(
    feature_used='logmelfb', duration_range=(2, 6), epoch=462)


# Loading Audio Files to the Class
model_trainer.load_audio_files()


# Extracting Features and Constructing Model Input and Target Values
model_trainer.construct_inputs()
model_trainer.construct_targets()


# Spliting Training and Validation Sets
model_trainer.split_train_validation_sets()
model_trainer.get_input_target_shapes()

# ## STACKED LSTM LAYER MODEL


# Defining Input parameters for model and selecting model to use
model_trainer.define_model_input_params()
model_trainer.create_train_predict_models(model_name='stacked_lstm')


# Training Selected Model
model_trainer.train_model(model_name='stacked-lstm-new', mlflow_experiment='STACKED LSTM Layers',
                          batch_size=70, optimizer='adam', learning_rate=0.001, early_stop=False)  # 0.01(best so far)


# Displaying Trained Model Loss Diagrams
# Loss Diagram (Type 1)
model_trainer.draw_result_plots('stacked_lstm_new', 1)
# LER Diagram (Type 2)
model_trainer.draw_result_plots('stacked_lstm_new', 2)

# Transcribing Validation Set
model_trainer.transcript_validation()

# Getting Sample Transcriptions Made by the Model
model_trainer.get_sample_trained_model_transcriptions(amount=10)


# # ## Bidirectional LSTM MODEL


# # Defining Input parameters for model and selecting model to use
# model_trainer.define_model_input_params()
# model_trainer.create_train_predict_models(model_name='bidirectional_lstm')


# # # Training Selected Model
# model_trainer.train_model(model_name='bidirectional_lstm', mlflow_experiment='BIDIRECTIONAL LSTM Layers',
#                           batch_size=140, optimizer='adam', learning_rate=0.01, early_stop=False)


# # Displaying Trained Model Loss Diagrams
# # Loss Diagram (Type 1)
# model_trainer.draw_result_plots('bidirectional_lstm', 1)
# # LER Diagram (Type 2)
# model_trainer.draw_result_plots('bidirectional_lstm', 2)


# # # Getting Sample Transcriptions Made by the Model
# model_trainer.get_sample_trained_model_transcriptions()

# # Getting Sample Transcriptions Made by the Model, checking if augmented files have different transcriptions
# model_trainer.get_sample_trained_model_transcriptions(check_augmentation=True)
