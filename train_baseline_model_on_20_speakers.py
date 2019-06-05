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


# These control how much data we _train_ on. We have 1047 minutes available per
# speaker in the train/dev set on disk, so setting this number higher than that
# is a no-op.
MINUTES_PER_SPEAKER = 150
# We have 20 speakers but can decrease this to train on just a subset.
NUM_SPEAKERS = 20

# Don't change these, they reflect what's on disk from the preprocessing stage.
NUM_ATTRIBUTES_PER_SAMPLE = 49 * 13 * 3
NUM_SAMPLES_PER_FILE = 120
DTYPE_ON_DISK = 'float32' # We used float32 to save disk space.

# |directory| is relative to the current directory.
def load_mfccs(directory, num_speakers, minutes_per_speaker):
  # In Keras, you want samples to be the rows, so the height of the matrix is the
  # number of samples.
  # Our samples are half second long, so multiply number of seconds * 2.
  num_samples = minutes_per_speaker * num_speakers * 60 * 2
  network_inputs = np.zeros((num_samples, NUM_ATTRIBUTES_PER_SAMPLE),
                           dtype=DTYPE_ON_DISK)
  labels = np.full((num_samples, num_speakers), -1)
  files_already_processed = 0
  for speaker_num in range(1, num_speakers + 1):
      file_glob = os.path.abspath(directory) + os.sep + "speaker_%d_*.npy" % speaker_num
      files = glob.glob(file_glob)
      assert len(files) > 0, "We found no files for speaker %d for glob %s" % (speaker_num, file_glob)
      assert len(files) >= minutes_per_speaker, (len(files), minutes_per_speaker, speaker_num, file_glob)
      random.shuffle(files)
      files = files[:minutes_per_speaker]
      assert len(files) <= minutes_per_speaker, "We have %d files" % len(files)
      for npy in files:
          these_samples = np.load(npy)
          assert(these_samples.shape == (NUM_ATTRIBUTES_PER_SAMPLE * NUM_SAMPLES_PER_FILE,)), these_samples.shape
          sample_start_index = files_already_processed * NUM_SAMPLES_PER_FILE
          network_inputs[sample_start_index:sample_start_index + NUM_SAMPLES_PER_FILE, :] = these_samples.reshape((NUM_SAMPLES_PER_FILE, -1))
          labels[sample_start_index:sample_start_index + NUM_SAMPLES_PER_FILE, :] = keras.utils.to_categorical(speaker_num - 1, num_classes=num_speakers)
          files_already_processed += 1

  return (network_inputs, labels)

train_dev_set, train_dev_labels = load_mfccs("data_np_save/train_dev", NUM_SPEAKERS, MINUTES_PER_SPEAKER)

# Cut down to only first derivative of MFCC
#train_dev_set = train_dev_set[:, 0:(49 * 13 * 2)]


X_train, X_dev, y_train, y_dev = train_test_split(train_dev_set,
                                                  train_dev_labels,
                                                  test_size = 0.1,
                                                  random_state = 35,
                                                  shuffle = True)

print ("%d seconds to load the data from disk" % (time.time() - start_time))

test_set_inputs, test_set_labels = load_mfccs("noisy_mfccs", NUM_SPEAKERS, 30)
#%%

#Keras
model = Sequential()

#hidden layers
# 250/200/200 got us 72%/57% on noisy data with 2.5 hours per speaker
# 350/200/200/100 got us 73%/63% on noisy data with 5 hours per speaker
# So making the network bigger didn't really help much.
model.add(layers.Dense(350, activation='relu', input_dim=train_dev_set.shape[1]))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(NUM_SPEAKERS, activation='softmax'))

adam_optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0)
model.compile(optimizer=adam_optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
#display(IPython.display.Image('test_keras_plot_model.png'))
print(model.summary())
# I don't know where 140240419526080 in the picture came from

start_time = time.time()
#with tf.device('/cpu:0'):
history_object = model.fit(X_train, y_train, epochs=40, batch_size=1024,
                           verbose=2, shuffle=True, validation_data=(X_dev, y_dev))
print ("%d seconds to train the model" % (time.time() - start_time))

model.save("baseline_model.h5")

generate_confusion_matrix(model, test_set_inputs, test_set_labels)

#%%
# Plot training & validation accuracy values
plt.plot(history_object.history['acc'])
plt.plot(history_object.history['val_acc'])
plt.title('Fully Connected Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('baseline_accuracy.png', bbox_inches='tight')
plt.close()
#plt.show()

# Plot training & validation loss values
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Fully Connected Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('baseline_loss.png', bbox_inches='tight')
#plt.show()
