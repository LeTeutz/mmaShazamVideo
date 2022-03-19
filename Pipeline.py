import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

import Utility as util
import Processing as proc


def start(file_path):
    frames = util.getFrames(file_path)

    # Pipeline:
    # --1. Process and rotate all the images
    frames = proc.findScreensInFrames(frames)

    util.displayFrames(frames)
