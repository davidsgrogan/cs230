#!/usr/bin/env python3

import scipy.io.wavfile
from python_speech_features import mfcc, delta
import numpy as np
import math

np.random.seed(1234)

files = [ 'cousinhenry_01_trollope_8khz.wav',
'siegeofcorinth_2_byron_8khz.wav',
'upperroom_16_ryle_8khz.wav',
'vorst_14_machiavelli_8khz.wav',
 ]

height_of_one_training_example = 49 * 13 * 2 + 1

label = 0
all_examples = []
for one_file in files:
  label += 1
  rate, data = scipy.io.wavfile.read(one_file)
  total_length_of_wave = data.shape[0]
  print ("just read file number %d which contains %d audio samples and is named %s Now analying it:" % (label, total_length_of_wave, one_file))
  assert rate == 8000, "rate was %d" % rate

  half_second_length = 4000
  start_index_of_half_second = 0
  num_training_example_in_this_file = 0
  while total_length_of_wave - start_index_of_half_second >= half_second_length:
    num_training_example_in_this_file += 1
    if num_training_example_in_this_file % 500 == 0:
      print ("\t analyzing training sample number %d" % num_training_example_in_this_file)

    this_training_example_raw = data[start_index_of_half_second:start_index_of_half_second + half_second_length]
    start_index_of_half_second += half_second_length
    assert len(this_training_example_raw) == 4000, len(this_training_example_raw)
    mfccs = mfcc(this_training_example_raw, 8000)
    assert mfccs.shape == (49, 13), mfccs.shape
    # Alfredo used 2 here, and changing it doesn't change the output size.
    first_derivative = delta(mfccs, 2)
    assert first_derivative.shape == (49, 13), first_derivative.shape
    all_examples.extend(mfccs.flatten().tolist())
    all_examples.extend(first_derivative.flatten().tolist())
    all_examples.append(label)
    assert len(all_examples) % height_of_one_training_example == 0, "num_training_example_in_this_file = %d" % num_training_example_in_this_file

all_examples_np = np.array(all_examples)
all_examples_np = all_examples_np.reshape((height_of_one_training_example, -1), order='F')

print ("all_examples_np.shape = %s, so we have %d training samples" % (all_examples_np.shape, all_examples_np.shape[1]))
assert all_examples_np[-1, 0] == 1, "make sure the last row labels the first column as belonging to file number 1 %s" % all_examples_np[-1, 0]

shuffled_examples = all_examples_np.T
np.random.shuffle(shuffled_examples)
shuffled_examples = shuffled_examples.T

training_pct = 0.8

number_of_training_examples = int(math.ceil(all_examples_np.shape[1] * training_pct))

X_train = shuffled_examples[0:-1, 0:number_of_training_examples]
Y_train = shuffled_examples[-1:, 0:number_of_training_examples]
X_dev   = shuffled_examples[0:-1, number_of_training_examples:]
Y_dev   = shuffled_examples[-1:, number_of_training_examples:]


# Xs are shape (number of input features, number of data points)
# Ys are shape (1, number of data points)
# The labels in Y are an integer corresponding to the speaker number.
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)
