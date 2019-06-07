#!/usr/bin/env python3
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras import optimizers
from keras import layers
from keras.models import Sequential
import keras.utils
from sklearn.model_selection import train_test_split
from confusion_matrix import generate_confusion_matrix

import matplotlib.pyplot as plt
import IPython.display

from pydub import AudioSegment
import librosa

import glob
import time
import random
import math
import os
import sys

# This workaround is only needed on my mac. It won't harm other computers though.
os.environ["PATH"] += os.pathsep + \
    '/Users/dgrogan/anaconda3/pkgs/graphviz-2.40.1-hefbbd9a_2/bin'


np.random.seed(1268)
# 85.8%, then 82% after restarting kernel, then 87% after resetting again
random.seed(1889)
# After setting sklearn seed, 84.7% then 85.2%, 84.8% <-- these were with batch size 128
# Then using batch size of 256, finished in 55% of the time, but got only 84%
# Then after changing batch size to 512, finished in 1/3 the time of 128, but only got 80%
#  - then 84.4%, 83%
# Then after changing batch size to 1024, only got 81% twice in a row, finished
# in 2/3 the time of batch size of 512
# random.seed(183) # 86.7%
# random.seed(1832) # 85%
# random.seed(18932) # 85%

# We get a validation accuracy of 54, 55, or 56% depending on the random state
# for 20 speakers with 60 minutes each.
# We get a validation of 40-45% for 20 speakers with 6 minutes each including
# both derivatives.
# We get validation of 36-48% for 20 speakers, 6 minutes each, including only
# the first derivative.


start_time = time.time()

# This cell reads the pre-processed audio features from disk and stuffs it into
# an np.array for Keras.

# We have
# 20 speakers
# 1102 files per speaker
# 120 samples per file (1 minute of audio per file)
# For a total 2,644,800 training examples

# These control how much data we _train_ on. We have 1047 minutes available per
# speaker in the train/dev set on disk, so setting this number higher than that
# is a no-op.
MINUTES_PER_SPEAKER = 120
# We have 20 speakers but can decrease this to train on just a subset.
NUM_SPEAKERS = 20

# 22050 * 60 * 60 * 2 bytes = 150 MB per hour per speaker


top_20 = [
    # This is speaker 1 and the list is always in this order.
    "nowitcanbetold",
    "unspokensermons",
    "littlewomen",
    "artcookerymadeplaineasy1784",
    "historymathematics",
    "mysticalcityofgod1",
    "newtestament",
    "uncletomscabin",
    "seapower",
    "wildwales",
    "originofspecies",
    "mysteries",
    "1001nacht2",
    "rkopis",
    "innocentsabroad",
    "woutertjepieterse",
    "geschichtedespeloponnesischenkriegs",
    "vanityfair",
    "jerusalemrevelationsr",
    "worldenglishbible",
]

# We use half second per sample.
NUM_SAMPLES_PER_SPEAKER = MINUTES_PER_SPEAKER * 60 * 2
NUM_SAMPLES = NUM_SPEAKERS * NUM_SAMPLES_PER_SPEAKER
SAMPLE_RATE = 22050
HALF_SECOND_OF_SAMPLES = int(SAMPLE_RATE / 2)

# %%
network_inputs = np.zeros((NUM_SAMPLES, HALF_SECOND_OF_SAMPLES))
labels = np.full((NUM_SAMPLES, NUM_SPEAKERS), -1)


samples_so_far_total = 0
for speaker_id, file_prefix in enumerate(top_20, 1):
    if speaker_id > NUM_SPEAKERS:
        break
    this_speaker_glob = os.path.abspath('noisy_top_20/%s_*' % file_prefix)
    list_of_mp3s_for_one_speaker = sorted(glob.glob(this_speaker_glob))
    assert len(list_of_mp3s_for_one_speaker) > 0, this_speaker_glob
    speaker_start_time = time.time()
    samples_so_far_for_this_speaker = 0
    for mp3 in list_of_mp3s_for_one_speaker:
        print("loading ", mp3)
        audio_time_series, sampling_rate = librosa.core.load(mp3, sr=None)
        assert sampling_rate == SAMPLE_RATE, (
            "frame_rate was %d" % sampling_rate)
        assert len(
            audio_time_series.shape) == 1, "It wasn't mono: %s" % audio_time_series.shape
        num_audio_samples_we_use = HALF_SECOND_OF_SAMPLES * \
            int(audio_time_series.shape[0] / HALF_SECOND_OF_SAMPLES)
        training_samples_from_this_file = audio_time_series[:num_audio_samples_we_use].reshape(
            (-1, HALF_SECOND_OF_SAMPLES))
        number_training_samples_we_have = training_samples_from_this_file.shape[0]
        number_training_samples_we_need = NUM_SAMPLES_PER_SPEAKER - \
            samples_so_far_for_this_speaker
        if number_training_samples_we_have > number_training_samples_we_need:
            training_samples_from_this_file = training_samples_from_this_file[
                :number_training_samples_we_need, :]
        network_inputs[samples_so_far_total:samples_so_far_total +
                       training_samples_from_this_file.shape[0], :] = training_samples_from_this_file
        labels[samples_so_far_total:samples_so_far_total+training_samples_from_this_file.shape[0],
               :] = keras.utils.to_categorical(speaker_id - 1, num_classes=NUM_SPEAKERS)
        samples_so_far_total += training_samples_from_this_file.shape[0]
        samples_so_far_for_this_speaker += training_samples_from_this_file.shape[0]
        if samples_so_far_for_this_speaker >= NUM_SAMPLES_PER_SPEAKER:
            assert samples_so_far_for_this_speaker == NUM_SAMPLES_PER_SPEAKER, samples_so_far_for_this_speaker
            break

print(network_inputs.shape)
print(network_inputs)
print(labels.shape)
print(labels)

# Conv1D expects there to be existing channels so add another dimension to the
# shape.
train_dev_set = np.expand_dims(network_inputs, axis=-1)
train_dev_labels = labels

print("train_dev_set.shape =", train_dev_set.shape)

# %%

X_train, X_dev, y_train, y_dev = train_test_split(train_dev_set,
                                                  train_dev_labels,
                                                  test_size=0.1,
                                                  random_state=35,
                                                  shuffle=True)

print("%d seconds to load the data from disk" % (time.time() - start_time))
# %%

# https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43960.pdf
# They use a 1-D CNN with P filters on the raw input signal and found that each
# filter roughly corresponded to a frequency. !!

"""
Pasted from the paper above:
First, we
take a small window of the raw waveform of length M samples,
and convolve the raw waveform with a set of P filters. If we
assume each convolutional filter has length N and we stride the
convolutional filter by 1, the output from the convolution will
be (M − N + 1) × P in time × frequency. Next, we pool the
filterbank output in time (thereby discarding short term phase
information), over the entire time length of the output signal,
to produce 1 × P outputs. Finally, we apply a rectified nonlinearity, followed
by a stabilized logarithm compression, to produce a frame-level feature vector
at time t
"""


"""
data_format: A string, one of channels_last (default) or channels_first. The
ordering of the dimensions in the inputs.  channels_last corresponds to inputs
with shape  (batch, steps, features) while channels_first corresponds to inputs
with shape  (batch, features, steps).
"""

# Keras
model = Sequential()
model.add(layers.Conv1D(filters=40, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal',
                        input_shape=(train_dev_set.shape[1], 1)))
model.add(layers.Conv1D(filters=50, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=80, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=80, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=100, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Conv1D(filters=120, kernel_size=3, strides=2,
                        activation='relu', kernel_initializer='glorot_normal'))
model.add(layers.Flatten())
model.add(layers.Dense(NUM_SPEAKERS, activation='softmax'))

adam_optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0)
model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

keras.utils.plot_model(
    model, to_file='test_keras_plot_model.png', show_shapes=True)
# display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())

start_time = time.time()
# with tf.device('/cpu:0'):
# The baseline model has input size 1911 so we could use batch size 1024.
# But the CNN has input size 11025, so we have to reduce the batch_size or the
# GPU runs out of memory.
history_object = model.fit(X_train, y_train, epochs=55, batch_size=32,
                           verbose=2, shuffle=True, validation_data=(X_dev, y_dev))
print("%d seconds to train the model" % (time.time() - start_time))

model.save("baseline_model.h5")

#generate_confusion_matrix(model, test_set_inputs, test_set_labels)

# %%
# Plot training & validation accuracy values
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('CNN Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnn_accuracy.png', bbox_inches='tight')
plt.close()
# plt.show()

# Plot training & validation loss values
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('cnn_loss.png', bbox_inches='tight')
# plt.show()
