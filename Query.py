#!/usr/bin/env python
import argparse
import video_search
import numpy as np
import cv2
import glob
from scipy.io import wavfile
from video_tools import *
import feature_extraction as ft    
import sys
import os
from video_features import *


""" 
parser = argparse.ArgumentParser(description="Video Query tool")
parser.add_argument("training_set", help="Path to training videos and wav files")
parser.add_argument("query", help="query video")
parser.add_argument("-s", help="Timestamp for start of query in seconds", default=0.0)
parser.add_argument("-e", help="Timestamp for end of query in seconds", default=0.0)
parser.add_argument("-f", help="Select features "+str(features)+" for the query ", default='colorhists')
args = parser.parse_args()
"""


def sliding_window(q_duration, frame_rate, x, w, compare_func):
    """ Slide window w over signal x.

        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    # Compute the score for each frame
    best_matches = []
    wl = len(w) 
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        frame_number = i
        best_matches.append((diff, frame_number))

    # Sort the frames increasingly in terms of difference
    best_matches.sort(key = lambda x: x[0])
    
    # Iterate through all frames
    last_frame = -np.inf
    ans = []
    shift_value = 10
    for i in range(len(best_matches)):
        if len(ans) == 3:
            break
        # If current frame can be added to the answer list, do so
        (score, frame) = best_matches[i]
        if frame - last_frame >= q_duration * frame_rate:
            ans.append((frame, score))
            i += shift_value - 1
            last_frame = frame # Update the last frame added to the list

    return ans


def euclidean_norm_mean(x,y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)


def euclidean_norm(x,y):
    return np.linalg.norm(x-y)


features = ['colorhists', 'tempdiffs', 'audiopowers', 'mfccs', 'colorhistdiffs']

def queryDatabase(file_path, frames, start, end, training_set, feature):
    frame_count = get_frame_count(file_path) + 1
    frame_rate = get_frame_rate(file_path)
    q_duration = float(start) - float(end)
    q_total = get_duration(file_path)

    if not float(start) < float(end) < q_total:
        print('Timestamp for end of query set to:', q_duration)
        end = q_total
    
    # Load audio data if necessary
    if feature == features[2] or feature == features[3]:
        filename, fileExtension = os.path.splitext(file_path)
        audio = filename + '.wav'
        fs, wav_data = wavfile.read(audio)

    query_features = []
    prev_frame = None
    prev_colorhist = None
    frame_nbr = int(start)*frame_rate
    for frame in frames:
        if frame is None:
            break

        if feature == features[0]:
            h = ft.colorhist(frame)
        elif feature == features[1]:
            h = temporal_diff(prev_frame, frame, 10)
        elif feature == features[2] or feature == features[3]:
            audio_frame = frame_to_audio(frame_nbr, frame_rate, fs, wav_data)
            if feature == features[2]:
                h = np.mean(audio_frame**2)
            elif feature == features[3]:
                h, mspec, spec = ft.extract_mfcc(audio_frame, fs)
        elif feature == features[4]:
            colorhist = ft.colorhist(frame)
            h = colorhist_diff(prev_colorhist, colorhist)
            prev_colorhist = colorhist
            
        if h is not None:
            query_features.append(h)
        prev_frame = frame
        frame_nbr += 1

    # Compare with database
    video_types = ('*.mp4', '*.MP4', '*.avi')
    audio_types = ('*.wav', '*.WAV')

    # Grab all video file names
    video_list = []
    for type_ in video_types:
        files = training_set + '/' +  type_
        video_list.extend(glob.glob(files))    

    db_name = '../../Code/db/video_database.db'
    search = video_search.Searcher(db_name)

    # Loop over all videos in the database and compare frame by frame
    for video in video_list:
        print(video)
        if get_duration(video) < q_duration:
            print(get_duration(video), q_duration)
            print('Error: query is longer than database video')
            continue

        w = np.array(query_features)
        if feature == features[0]:
            x = search.get_colorhists_for(video)
            scores = sliding_window(q_duration, frame_rate, x, w, euclidean_norm_mean)
        elif feature == features[1]:
            x = search.get_temporaldiffs_for(video)
            scores = sliding_window(q_duration, frame_rate, x, w, euclidean_norm)
        elif feature == features[2]:
            x = search.get_audiopowers_for(video)
            scores = sliding_window(q_duration, frame_rate, x, w, euclidean_norm)
        elif feature == features[3]:
            x = search.get_mfccs_for(video)
            #frame, score = sliding_window(x,w, euclidean_norm_mean)
            print(x.shape)
            availableLength= min(x.shape[1],w.shape[1])
            scores = sliding_window(q_duration, frame_rate, x[:,:availableLength,:], w[:,:availableLength,:], euclidean_norm_mean)
        elif feature == features[4]:
            x = search.get_chdiffs_for(video)
            scores = sliding_window(q_duration, frame_rate, x, w, euclidean_norm)
        
        
        print('Best matches at:')
        for i in range(len(scores)):
            (frame, score) = scores[i]
            print(str(i + 1) + '.', frame/frame_rate, 'seconds, with score of:', score)
        print('')
