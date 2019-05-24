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
# Throw out "the" because it describes two audio books.
top_20.remove("the");
for title in top_20:
    print(title, total_durations[title])
    #print(title)

duration_of_least = total_durations[top_20[-1]]

print ("\nThe speaker with least data has %d seconds. So stop there for all speakers."
       % duration_of_least)

# nowitcanbetold has 68431 seconds. 68400 is exactly 19 hours, so just use that?
