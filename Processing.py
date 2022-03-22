import cv2
import imutils
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import random as rng


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


def findScreensInFramesImprovedCanny(frames):
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


def findScreensInFramesCanny(frames):
    result_frames = []
    screens = []

    for i in range(len(frames)):
        frame = frames[i]
        H, W, C = frame.shape

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 5 / 3, cv2.BORDER_DEFAULT)
        # frame = cv2.Canny(frame, 60, 120)

        v = np.median(frame)
        # v /= 3
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        frame = cv2.Canny(frame, lower, upper)
        frame = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

        keypoints = cv2.findContours(frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        candidates = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            (x, y), (h, w), R = cv2.minAreaRect(approx)

            if len(approx) != 4:
                continue

            # If the contour is less than 10% of the area of the frame -> discarded
            if cv2.contourArea(approx) / (W * H) < 0.10:
                continue

            aspectratio = float(max(W, H)) / min(W, H)
            if not (1 <= aspectratio <= 3):
                continue

            candidates.append(approx)
            # color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            # cv2.drawContours(frame, [approx], 0, color, 2)

        if len(candidates) > 0:
            candidates = sorted(candidates, key=cv2.contourArea)
            color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
            cv2.drawContours(frame, [candidates[0]], 0, color, 2)
            screen = candidates[0]
        else:
            screen = None

        result_frames.append(frame)
        screens.append(screen)

    return result_frames, screens


def findScreensInFramesBKGSubst(frames):
    result_frames = []

    fgbg = cv2.createBackgroundSubtractorMOG2()
    for frame in frames:
        fgmask = fgbg.apply(frame)
        result_frames.append(fgmask)

    return result_frames


def bringScreensToFront(frames, screens):
    output_screens = []

    for i in range(len(frames)):
        frame = frames[i]
        screen = screens[i]

        if screen is None:
            output_screens.append(np.zeros((frame.shape[1], frame.shape[0], 3), dtype=np.uint8))
            continue
        
        # Compute the minimum area rotated rectangle of the screen
        (X, Y), (W, H), R = cv2.minAreaRect(screen)
        X = int(X)
        Y = int(Y)
        W = int(W)
        H = int(H)

        # If the screen is "mostly vertical", then consider it a phone screen?
        if R > 45:
            R -= 90
            W, H = H, W

        if R < -45:
            R += 90
            W, H = H, W

        left_upper_corner = [int(X - W / 2), int(Y - H / 2)]
        left_down_corner = [int(X - W / 2), int(Y + H / 2)]
        right_upper_corner = [int(X + W / 2), int(Y - H / 2)]
        right_down_corner = [int(X + W / 2), int(Y + H / 2)]

        # Get the transformation matrix from a rotated screen to a screen with edges parallel to the axis 
        pts1 = np.float32([left_upper_corner, left_down_corner, right_upper_corner, right_down_corner])
        pts2 = np.float32([[0, 0], [0, H], [W, 0], [W, H]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the transformation to the input
        dst = cv2.warpPerspective(frame, M, (W, H))
        output_screens.append(dst)

        # x, y, w, h = cv2.boundingRect(screen)

        # x = int(x)
        # y = int(y)
        # w = int(w)
        # h = int(h)
        # X = int(X)
        # Y = int(Y)
        # W = int(W)
        # H = int(H)

        # if not (R < 45):
        #     R = R - 90
        #     W, H = H, W
        #
        # matrix = cv2.getRotationMatrix2D((X, Y), R, 1.0)
        # output = cv2.warpAffine(frame, matrix, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        #
        # newX = int(X - W / 2)
        # newY = int(Y - H / 2)
        #
        # cropped_output = output[newY:newY + int(H), newX:newX + int(W)]
        #
        # print(R)
        # output_screens.append(cropped_output)

    return output_screens
