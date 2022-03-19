import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import time
import Pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='wtl_noise.mp4')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    file_path = args.file_path
    start_time = time.time()

    Pipeline.start(file_path)

    print("--- %s seconds ---" % (time.time() - start_time))
