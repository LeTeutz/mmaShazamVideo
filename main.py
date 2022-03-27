import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import time
import Pipeline
import Utility as util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='Videos/wtl_noise.mp4')
    parser.add_argument('--sample_frequency', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--training_set', type=str, default='~/mma-lab/Code/db/video_database.db')
    parser.add_argument('--feature', type=str, default='colorhists')

    # Parse the arguments from the input command
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Extract the parsed arguments
    args = get_args()
    file_path = args.file_path
    sample_frequency = args.sample_frequency
    start = args.start
    end = args.end
    training_set = args.training_set
    feature = args.feature

    start_time = time.time() 

    Pipeline.start(file_path, sample_frequency, start, end, training_set, feature) # Starting the pipeline

    print("--- %s seconds ---" % (time.time() - start_time)) # Tracking the total processing time
