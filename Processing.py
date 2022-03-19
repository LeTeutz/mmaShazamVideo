import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math


class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


def findScreensInFrames(frames):
    cv2.dnn_registerLayer('Crop', CropLayer)
    net = cv2.dnn.readNet('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')
    result_frames = []

    # TAKES TOO LONG TO RUN

    for i in range(len(frames)):
        frame = frames[i]
        h, w, c = frame.shape

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.GaussianBlur(frame, (5, 5), 5/3, cv2.BORDER_DEFAULT)

        inp = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(w, h),
                                    swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()
        out = out[0, 0]
        out = cv2.resize(out, (frame.shape[1], frame.shape[0]))
        out = 255 * out
        out = out.astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        result_frames.append(out)

    return result_frames
