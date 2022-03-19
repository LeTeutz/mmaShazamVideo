import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math


def getFrames(path):
    frames = []
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()
    return frames


def displayFrames(frames):
    for i in range(len(frames)):
        cv2.imshow('Frame', frames[i])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
