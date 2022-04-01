import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

import Utility as util
import Processing as proc
import Query as query


def start(file_path, sample_frequency, start, end, training_set, feature):
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

    # --5. Query the database and print answer
    final_answers = query.queryDatabase(file_path, frames, start, end, training_set, feature)
    print("Best answers can be found in videos:")
    for i in range(5):
        (best_video, best_score) = final_answers[i]
        print(str(best_video) + " with score " + str(best_score))
    print('')

<<<<<<< HEAD
    return final_answers
    #util.displayFrames(frames)
=======
    # util.displayFrames(frames)
>>>>>>> d14d612ce8bafc100f8b4f4d151967ee27f91ca5
