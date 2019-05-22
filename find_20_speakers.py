#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:20:50 2019

@author: dgrogan
"""

import glob
import librosa
import os
from collections import defaultdict

#%%
#mp3_glob = '/usr/local/google/home/mse/voice-classification/data_collection/vanityfair_24_thackeray_64kb.mp3'
#mp3_glob = '/usr/local/google/home/mse/voice-classification/data_collection/theologicopoliticaltreatise_04_spinoza_64kb.mp3'
#mp3_glob = '/usr/local/google/home/mse/voice-classification/data_collection/chimneysmoke*.mp3'
mp3_glob = '/usr/local/google/home/mse/voice-classification/data_collection/*.mp3'

list_of_mp3s = glob.glob(mp3_glob)

total_durations = defaultdict(int)

for file in list_of_mp3s:
    prefix = os.path.basename(file).split('_')[0]
    assert len(prefix) > 0
    assert not prefix.endswith("mp3")
    print (prefix)
    total_durations[prefix] += librosa.core.get_duration(filename=file)

#with open('durations_dict.txt', 'w') as output_file:
    #print(dict(total_durations), file=output_file)
