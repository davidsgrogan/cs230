#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:02:49 2019
"""

from scipy import signal
import scipy.io.wavfile
from python_speech_features import mfcc, delta

rate, data = scipy.io.wavfile.read('vorst_14_machiavelli_8khz.wav')
#rate, data = scipy.io.wavfile.read('upperroom_16_ryle_8khz.wav')
print("number of samples", data.shape[0])
print("rate = %d" % rate)

freqs, times, specgram = signal.spectrogram(data[0:4000])
print("specgram.shape", specgram.shape)

# For data 3008270  long, specgram.shape is (129, 13429)
# For data 36,554,719 long, specgram.shape is (129, 163190) = 21,051,510
# ^ Seems to be window-independent
# Shrinks by a factor of 1.74

mfccs = mfcc(data[0:4000], 8000)
print ("mfccs.shape", mfccs.shape)

# Alfredo used 2 here:
first_derivative = delta(mfccs, 2)
print ("first_derivative.shape", first_derivative.shape)

# For 36,554,719 samples we get (456933, 13) mfcc values = 5,940,129 numbers
# But for half-second aka 4000 samples, it's 49 * 13 = 637.
# ^ independent of numfilt, but numcep is the last axis
# Shrinks by a factor of 6.15, but if we use both derivatives, that's down to 2.
