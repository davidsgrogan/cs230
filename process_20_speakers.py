#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:20:50 2019

@author: dgrogan
"""

import glob
import librosa
import os
import time
from python_speech_features import mfcc, delta
import pickle
import numpy as np

def write_to_file(speaker_id, one_minute_of_examples, file_num, minute_num):
  np_array = np.array(one_minute_of_examples, dtype="float32")
  filename = 'data_np_save/speaker_%d_file_%d_cumulative_minute_%d' % (speaker_id, file_num, minute_num)
  print ("saving", filename)
  np.save(filename, np_array)

speaker_id = 1
top_20 = [
# "nowitcanbetold", I processed this one out of band and labeled it speaker 1
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
for file_prefix in top_20:
    speaker_id += 1
    this_speaker_glob = ('/usr/local/google/home/dgrogan/top20_mp3/%s_*.mp3'
                         % file_prefix)
    list_of_mp3s_for_one_speaker = sorted(glob.glob(this_speaker_glob))
    speaker_start_time = time.time()
    minute_num = 0
    file_num = 0
    for mp3 in list_of_mp3s_for_one_speaker:
        file_num += 1
        file_start_time = time.time()
        print ("loading ", mp3)
        audio_time_series, sampling_rate = librosa.core.load(mp3, sr=None)
        print ("\t \-- that had sampling_rate of %d and time_series shape of %s" %
               (sampling_rate, audio_time_series.shape))
        print ("librosa.core.load took %d seconds" % (time.time() - file_start_time))
        half_second_length = int(sampling_rate / 2)
        total_length_of_wave = audio_time_series.shape[0]
        start_index_of_half_second = 0
        one_minute_of_examples = []
        while total_length_of_wave - start_index_of_half_second >= (120 * half_second_length):
#            print (start_index_of_half_second, start_index_of_half_second + half_second_length)
            this_training_example_raw = audio_time_series[start_index_of_half_second:start_index_of_half_second + half_second_length]
            start_index_of_half_second += half_second_length
            # 552 is 22050 * 0.025. These mp3s aren't downsampled like the wavs
            # were so there are more samples per half-second than the default
            # number of 512 fft.
            mfccs = mfcc(this_training_example_raw, samplerate=sampling_rate, nfft=552)
            assert mfccs.shape == (49, 13), mfccs.shape
            first_derivative = delta(mfccs, 2)
            assert first_derivative.shape == (49, 13), first_derivative.shape
            second_derivative = delta(first_derivative, 2)
            assert second_derivative.shape == (49, 13), second_derivative.shape
            one_minute_of_examples.extend(np.ravel(mfccs))
            one_minute_of_examples.extend(np.ravel(first_derivative))
            one_minute_of_examples.extend(np.ravel(second_derivative))
            assert len(one_minute_of_examples) % (49 * 13 * 3) == 0, len(one_minute_of_examples)
            if (len(one_minute_of_examples) == 49 * 13 * 3 * 120):
                write_to_file(speaker_id, one_minute_of_examples, file_num, minute_num)
                minute_num += 1
                if minute_num == 1106:
                    break
                one_minute_of_examples = []
        if minute_num == 1106:
            break
        print ("that file took %d seconds" % (time.time() - file_start_time))
    print ("that speaker took %d seconds" % (time.time() - speaker_start_time))
