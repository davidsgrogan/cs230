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

with open('durations_dict.txt', 'r') as content_file:
    content = content_file.read()

total_durations = eval(content)
print(total_durations)

sorted_by_duration = sorted(total_durations, key=lambda x: total_durations[x], reverse=True)

top_20 = sorted_by_duration[0:21]
# Throw out "the"
top_20.remove("the");
for title in top_20:
    print(title, total_durations[title])
    #print(title)

duration_of_least = total_durations[top_20[-1]]

print ("\nThe speaker with least data has %d seconds. So stop there for all speakers."
       % duration_of_least)

# nowitcanbetold has 68431 seconds. 68400 is exactly 19 hours, so just use that?

def write_to_file(speaker_id, one_minute_of_examples, file_num, minute_num):
    with open('data_np_save/speaker_%d_file_%d_cumulative_minute_%d.pkl' % (speaker_id, file_num, minute_num), 'wb') as output_file:
        pickle.dump(one_minute_of_examples, output_file)

speaker_id = 0
top_20 = ["nowitcanbetold"]
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
            print (start_index_of_half_second, start_index_of_half_second + half_second_length)
            this_training_example_raw = audio_time_series[start_index_of_half_second:start_index_of_half_second + half_second_length]
            start_index_of_half_second += half_second_length
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
                one_minute_of_examples = []
        print ("that file took %d seconds" % (time.time() - file_start_time))
    print ("that speaker took %d seconds" % (time.time() - speaker_start_time))
