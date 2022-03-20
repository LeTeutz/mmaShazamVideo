import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

import Utility as util
import Processing as proc


def start(file_path, sample_frequency):
    # frames = util.getFrames(file_path)
    # frames = [frames[i] for i in range(len(frames)) if i % sample_frequency == 0]

    # Pipeline:
    # --1. Stabilize Video
    frames = util.stabilizeVideo(file_path)
    # --2. Sample the video at a certain frequency
    frames = [frames[i] for i in range(len(frames)) if i % sample_frequency == 0]
    # --3. Process and rotate all the images
    canny_frames, screens = proc.findScreensInFramesCanny(frames)
    # --4. Crop the video to show just the screen, with rotation adjusted
    frames = proc.bringScreensToFront(frames, screens)

    util.displayFrames(frames)
