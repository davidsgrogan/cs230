#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: dgrogan
"""

import glob
import os
import time
import numpy as np
import random
import multiprocessing
import sys

from pydub import AudioSegment

def do_one_speaker(file_prefix):
    this_speaker_glob = os.path.abspath('top20_mp3/%s_*.mp3' % file_prefix)
    list_of_mp3s_for_one_speaker = sorted(glob.glob(this_speaker_glob))
    assert len(list_of_mp3s_for_one_speaker) > 10, "%s had %d mp3s" % (
        this_speaker_glob, len(list_of_mp3s_for_one_speaker))
    speaker_start_time = time.time()
    # If we run 1300 minutes of mp3s, we'll surely get 1101 minutes from them.
    length_in_seconds_max = 1300 * 60
    length_in_seconds_so_far = 0
    for mp3 in list_of_mp3s_for_one_speaker:
        if length_in_seconds_so_far > length_in_seconds_max:
            break
        file_start_time = time.time()
        print("loading ", mp3)

        audio_book = AudioSegment.from_file(mp3)
        assert audio_book.frame_rate == 22050, (
            "frame_rate was %d" % audio_book.frame_rate)
        assert audio_book.channels == 1, audio_book.channels
        print("\t \-- that had sampling_rate of %d and length in seconds of %s" %
              (audio_book.frame_rate, audio_book.duration_seconds))
        length_in_seconds_so_far += audio_book.duration_seconds
        print("pydub load took %d seconds" % (time.time() - file_start_time))

        noise_start = 0
        noise_duration_s = 20
        overlaid_segment = audio_book
        while noise_start < audio_book.duration_seconds - 1:
            noise_index = random.randint(0, len(noise_segments) - 1)
            overlaid_segment = overlaid_segment.overlay(
                    noise_segments[noise_index], position=1000*noise_start)
            noise_start += noise_duration_s
        filename = "noisy_top_20/" + os.path.basename(mp3) + "_NOISE.mp3"
        assert overlaid_segment.channels == 1, overlaid_segment.channels
        assert overlaid_segment.frame_rate == 22050
        overlaid_segment.export(filename).close()

        print("%s took %d seconds for loading and writing" %
              (filename, time.time() - file_start_time))
    print("%s took %d seconds" % (file_prefix, time.time() - speaker_start_time))


if __name__ == '__main__':

    def match_target_amplitude(sound):
        change_in_dBFS = -33 - sound.dBFS
        return sound.apply_gain(change_in_dBFS)


    speaker_id = 1
    top_20 = [
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
    
    noise_file_names = [
        "noise_sources/crowd-talking-8.mp3",
        "noise_sources/laptop-keyboard-1.wav",
        "noise_sources/plastic-crumple-1.mp3",
        # "/usr/local/google/home/dgrogan/cs230/noise_sources/car-reverse-1.mp3",
    ]
    
    noise_short_names = ["crowd", "laptop", "plastic"]
    
    noise_file_names = [os.path.abspath(noise_file_name) for noise_file_name in noise_file_names]
    noise_segments = [AudioSegment.from_file(i) for i in noise_file_names]
    assert noise_segments[0].duration_seconds > 0.5, "Does %s exist? It was only %s seconds long" %(noise_file_names[0], noise_segments[0].duration_seconds)
    noise_segments = [i.set_channels(1) for i in noise_segments]
    noise_segments = [i.set_frame_rate(22050) for i in noise_segments]
    noise_segments = [match_target_amplitude(i) for i in noise_segments]

#    for index, segment in enumerate(noise_segments):
#        segment.export(noise_short_names[index] + ".mp3").close()

    assert len(noise_file_names) == len(noise_segments)
    assert len(noise_segments) == len(noise_short_names)

#    do_one_speaker("nowitcanbetold")
#    sys.exit()

    try:
        os.makedirs("noisy_top_20")
    except OSError as e:
        print ("\nerror creating noisy_top_20/ does it already exist? If so you probably want to delete it\n")
        raise e

    print ("using %d noise segments" % len(noise_segments))
    jobs = []
    for file_prefix in top_20:
        p = multiprocessing.Process(target=do_one_speaker, args=(file_prefix,))
        jobs.append(p)
        p.start()
    print ("started everything")
    for p in jobs:
        p.join()
    print ("every job finished")
