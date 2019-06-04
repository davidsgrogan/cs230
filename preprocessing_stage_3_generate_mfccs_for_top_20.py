#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: dgrogan
"""

import glob
import librosa
import os
import time
from python_speech_features import mfcc, delta
import pickle
import numpy as np
import re

def write_to_file(speaker_id, one_minute_of_examples, file_num, minute_num, noise_id=None):
  np_array = np.array(one_minute_of_examples, dtype="float32")
  filename = 'noisy_mfccs/speaker_%d_file_%d_cumulative_minute_%d' % (speaker_id, file_num, minute_num)
  if noise_id != None:
    filename += "_" + noise_id
#  filename = 'data_np_save/speaker_%d_file_%d_cumulative_minute_%d' % (speaker_id, file_num, minute_num)
  print ("saving", filename)
  np.save(filename, np_array)


def process_one_speaker(speaker_id_file_prefix):
    speaker_id, file_prefix = speaker_id_file_prefix
    this_speaker_glob = os.path.abspath('noisy_top_20/%s_*' % file_prefix)
    list_of_mp3s_for_one_speaker = sorted(glob.glob(this_speaker_glob))
    assert len(list_of_mp3s_for_one_speaker) > 0, this_speaker_glob
    speaker_start_time = time.time()
    minute_num = 0
    file_num = 0
    for mp3 in list_of_mp3s_for_one_speaker:
        file_num += 1

        match = re.search(r"_(NOISE.+?)\.mp3$", mp3)
        assert match, mp3
        noise_id = match.group(1)

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
                write_to_file(speaker_id, one_minute_of_examples, file_num, minute_num, noise_id)
                minute_num += 1
                # The smallest audiobook counted by minutes, unspokensermons,
                # has 1101 minutes, so if we're about to process 1102, punt.
                if minute_num == 1102:
                    break
                one_minute_of_examples = []
        if minute_num == 1102:
            break
        print ("that file took %d seconds" % (time.time() - file_start_time))
    print ("that speaker took %d seconds" % (time.time() - speaker_start_time))

top_20 = [
"nowitcanbetold", # This is speaker 1 and the list is always in this order.
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

if __name__ == '__main__':

    try:
        os.makedirs("noisy_mfccs")
    except OSError as e:
        print ("\nerror creating noisy_mfccs/ does it already exist? If so you probably want to delete it\n")
        raise e

    from multiprocessing import Pool
    pool = Pool()
    print("starting %d processes" % pool._processes)

    try:
        result = pool.map(process_one_speaker, zip(range(1, len(top_20) + 1), top_20))
    except KeyboardInterrupt:
      pool.terminate()
      pool.join()
