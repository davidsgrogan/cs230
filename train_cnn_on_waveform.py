#!/usr/bin/env python3
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras import layers
from keras.models import Sequential
import keras.utils
import keras.optimizers
from sklearn.model_selection import train_test_split
from confusion_matrix import generate_confusion_matrix, generate_raw_dataset_from_mp3s_in_parallel

import matplotlib.pyplot as plt
import IPython.display

import time
import random
import os
import sys

# This workaround is only needed on my mac. It won't harm other computers though.
os.environ["PATH"] += os.pathsep + \
    '/Users/dgrogan/anaconda3/pkgs/graphviz-2.40.1-hefbbd9a_2/bin'


#np.random.seed(1268)
#random.seed(1889)

start_time = time.time()

# These control how much data we use for train/dev We have 1047 minutes
# available per speaker in the train/dev set on disk, so setting this number
# higher than that is a no-op.
MINUTES_PER_SPEAKER = 20
# We have 20 speakers but can decrease this to train on just a subset.
NUM_SPEAKERS = 20

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
(network_inputs, labels) = generate_raw_dataset_from_mp3s_in_parallel(
#        NUM_SPEAKERS, MINUTES_PER_SPEAKER, directory="top20_mp3")
        NUM_SPEAKERS, MINUTES_PER_SPEAKER, directory="noisy_top_20")

# Conv1D expects there to be existing channels so add another dimension to the
# shape.
train_dev_set = np.expand_dims(network_inputs, axis=-1)
train_dev_labels = labels

print (train_dev_set.shape)
print (train_dev_set)

print (train_dev_labels.shape)
print (train_dev_labels)

assert train_dev_set.shape == (NUM_SAMPLES, HALF_SECOND_OF_SAMPLES, 1), train_dev_set.shape
assert train_dev_labels.shape == (NUM_SAMPLES, NUM_SPEAKERS), train_dev_labels.shape

# TODO(dgrogan): We'll need a proper test set.
#(test_set_inputs, test_set_labels) = generate_raw_dataset_from_mp3s_in_parallel(
##        NUM_SPEAKERS, minutes_per_speaker=10, directory="top20_mp3")
#        NUM_SPEAKERS, minutes_per_speaker=10, directory="noisy_top_20")
#test_set_inputs = np.expand_dims(test_set_inputs, axis=-1)
#
#assert len(test_set_inputs.shape) == 3, test_set_inputs.shape

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

# Keras
model = Sequential()
model.add(layers.Conv1D(filters=40, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal',
                        input_shape=(X_train.shape[1], 1)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Conv1D(filters=50, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Conv1D(filters=60, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(layers.Conv1D(filters=80, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(layers.Conv1D(filters=100, kernel_size=3, strides=2,
                        activation=None, use_bias=False,
                        kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(keras.layers.MaxPooling1D(pool_size=2))

model.add(layers.Flatten())

model.add(layers.Dense(40, activation=None, use_bias=False,
                       kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))

model.add(layers.Dense(NUM_SPEAKERS, activation='softmax'))

adam_optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

keras.utils.plot_model(
    model, to_file='test_keras_plot_model.png', show_shapes=True)
# display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())
print("Just printed model summary")

start_time = time.time()
# with tf.device('/cpu:0'):
# The baseline model has input size 1911 so we could use batch size 1024.
# But the CNN has input size 11025, so we have to reduce the batch_size or the
# GPU runs out of memory.
tensor_board = keras.callbacks.TensorBoard(histogram_freq=1)
history_object = model.fit(X_train, y_train, epochs=25, batch_size=64,
                           verbose=2,
                           callbacks=[tensor_board],
                           shuffle=True,
                           validation_data=(X_dev, y_dev))
print("%d seconds to train the model" % (time.time() - start_time))

model.save("cnn_model.h5")

#generate_confusion_matrix(model, test_set_inputs, test_set_labels)
generate_confusion_matrix(model, X_dev, y_dev)

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
