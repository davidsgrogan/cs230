#!/usr/bin/env python3
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical
import keras.utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


import IPython.display

import glob
import time
import random
import math
import os

# This workaround is only needed on my mac. It won't harm other computers though.
os.environ["PATH"] += os.pathsep + '/Users/dgrogan/anaconda3/pkgs/graphviz-2.40.1-hefbbd9a_2/bin'


#%%
np.random.seed(1268)
random.seed(1889) # 85.8%, then 82% after restarting kernel, then 87% after resetting again
# After setting sklearn seed, 84.7% then 85.2%, 84.8% <-- these were with batch size 128
# Then using batch size of 256, finished in 55% of the time, but got only 84%
# Then after changing batch size to 512, finished in 1/3 the time of 128, but only got 80%
#  - then 84.4%, 83%
# Then after changing batch size to 1024, only got 81% twice in a row, finished
# in 2/3 the time of batch size of 512
#random.seed(183) # 86.7%
#random.seed(1832) # 85%
#random.seed(18932) # 85%

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

# I set aside 5% as the test set, which we'll only look at when we
# want an unbiased estimate of the prediction error of our final model. We can
# split our train/dev however we want as we go along.


# Limit the amount of data we train on. We have 1047 minutes available per
# speaker in the train/dev set on disk, so setting this number higher than that
# is a no-op.
MINUTES_PER_SPEAKER = 150
# We have a max of 20 speakers but can change this to train on just a subset.
NUM_SPEAKERS = 20

# Don't change these, they just reflect what's on disk.
NUM_ATTRIBUTES_PER_SAMPLE = 49 * 13 * 3
NUM_SAMPLES_PER_FILE = 120
DTYPE_ON_DISK = 'float32' # We used float32 to save disk space.

# In Keras, you want samples to be the rows, so the height of the matrix is the
# number of samples.
# Our samples are half second long, so multiply number of seconds * 2.
num_samples = MINUTES_PER_SPEAKER * NUM_SPEAKERS * 60 * 2
train_dev_set = np.zeros((num_samples, NUM_ATTRIBUTES_PER_SAMPLE),
                         dtype=DTYPE_ON_DISK)
train_dev_labels = np.full((num_samples, NUM_SPEAKERS), -1)


files_already_processed = 0
for speaker_num in range(1, NUM_SPEAKERS + 1):
    files = glob.glob("data_np_save/train_dev/speaker_%d_*.npy" % speaker_num)
    random.shuffle(files)
    files = files[:MINUTES_PER_SPEAKER]
    assert len(files) > 0, "We found no files for speaker %d when globbing from the directory %s" % (speaker_num, os.getcwd())
    assert len(files) <= MINUTES_PER_SPEAKER, "We have %d files" % len(files)
    for npy in files:
        these_samples = np.load(npy)
        assert(these_samples.shape == (NUM_ATTRIBUTES_PER_SAMPLE * NUM_SAMPLES_PER_FILE,)), these_samples.shape
        sample_start_index = files_already_processed * NUM_SAMPLES_PER_FILE
        train_dev_set[sample_start_index:sample_start_index + NUM_SAMPLES_PER_FILE, :] = these_samples.reshape((NUM_SAMPLES_PER_FILE, -1))
        train_dev_labels[sample_start_index:sample_start_index + NUM_SAMPLES_PER_FILE, :] = to_categorical(speaker_num - 1, num_classes=NUM_SPEAKERS)
        files_already_processed += 1

# Cut down to only first derivative of MFCC
#train_dev_set = train_dev_set[:, 0:(49 * 13 * 2)]


X_train, X_dev, y_train, y_dev = train_test_split(train_dev_set,
                                                  train_dev_labels,
                                                  test_size = 0.1,
                                                  random_state = 35,
                                                  shuffle = True)

print ("%d seconds to load the data from disk" % (time.time() - start_time))

#%%

#Keras
model = tf.keras.Sequential()

#hidden layers
model.add(layers.Dense(150, activation='relu', input_dim=train_dev_set.shape[1]))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(NUM_SPEAKERS, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())
# I don't know where 140240419526080 in the picture came from

start_time = time.time()
#with tf.device('/cpu:0'):
history_object = model.fit(X_train, y_train, epochs=60, batch_size=1024,
                           verbose=2, shuffle=True, validation_data=(X_dev, y_dev))
print ("%d seconds to train the model" % (time.time() - start_time))
#%%
# Plot training & validation accuracy values
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
