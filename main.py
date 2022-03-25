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
    parser.add_argument('--file_path', type=str, default='wtl_noise.mp4')
    parser.add_argument('--sample_frequency', type=int, default=5)

    video = util.readVideo(args.file_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=n_frames-1)
    parser.add_argument('--feature', type=str, default='colorhists')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    file_path = args.file_path
    sample_frequency = args.sample_frequency
    start = args.start
    end = args.end
    feature = args.feature

    start_time = time.time()

    Pipeline.start(file_path, sample_frequency, start, end, feature)

    print("--- %s seconds ---" % (time.time() - start_time))
