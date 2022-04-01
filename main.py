from cgi import test
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
    parser.add_argument('--file_path', type=str, default='Videos/f5.mp4')
    parser.add_argument('--sample_frequency', type=int, default=5)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--training_set', type=str, default='../../Videos')
    parser.add_argument('--feature', type=str, default='colorhists')

    # Parse the arguments from the input command
    args = parser.parse_args()
    return args


TESTING = False

if __name__ == '__main__':

    if TESTING is False:
        # Extract the parsed arguments
        args = get_args()
        file_path = args.file_path
        sample_frequency = args.sample_frequency
        start = args.start
        end = args.end
        training_set = args.training_set
        feature = args.feature

        start_time = time.time()

        try:
            Pipeline.start(file_path, sample_frequency, start, end, training_set, feature)  # Starting the pipeline
        except:
            print("Something wrong occured with the file")

        print("--- %s seconds ---" % (time.time() - start_time))  # Tracking the total processing time

    else:

        training_expectations = [
            # PC
            "./Videos/Asteroid_Discovery.mp4",
            "./Videos/British_Plugs_Are_Better.mp4",
            "./Videos/Danger_Humans.mp4",
            "./Videos/Google_Street_View_Race.mp4",
            # Laptop
            "./Videos/Kerbal_Space_Program1.mp4",
            "./Videos/supercoilsEN.mp4",
            "./Videos/TUDelft.mp4",
            "./Videos/TUDelft_Ambulance_Drone.mp4",
            # Phone
            "./Videos/How_Green_Screen_Worked_Before_Computers.mp4",
            "./Videos/How_YouTube_Stabilization_Works.mp4"
        ]

        testing_expectations = [
            # PC
            "./Videos/Civile_Techniek1.mp4",
            "./Videos/DeltaIV.mp4",
            "./Videos/Lemmings.mp4",
            "./Videos/Oversight.mp4",
            # Laptop
            "./Videos/Lets_Talk_About_Anstares_Launch_Failure.mp4",
            "./Videos/Looproute_TUDelft1.mp4",
            "./Videos/Open_Education_Week1.mp4",
            "./Videos/Thomas_Trueblood_Citation_Needed.mp4",
            # Phone
            "./Videos/The_Arctic_Winter_Games_Citation_Needed.mp4",
            "./Videos/Your_GPS_Shuts_Down_If_It_Goes_Too_Fast.mp4"
        ]
        # TRAINING

        sample_frequency = 1
        start = 0
        end = 10
        training_set = '../../Videos'
        feature = 'colorhist'

        for i in range(1, 11):
            file_path = "./Videos/t" + str(i) + ".mp4"
            try:
                final_answers = Pipeline.start(file_path, sample_frequency, start, end, training_set, feature)
                print("Training instance " + str(i) + ":\n\t")
                if training_expectations[i - 1] in final_answers:
                    print("MATCH for " + training_expectations[i - 1])
                else:
                    print("Expected " + training_expectations[i - 1] + " but got: " + str(final_answers))
            except:
                print("Something wrong occured")


        passed = 0
        for i in range(1, 11):
            file_path = "./Videos/f" + str(i) + ".mp4"
            try:
                final_answers = Pipeline.start(file_path, sample_frequency, start, end, training_set, feature)
                print("TEST " + str(i) + ":\n\t")
                if testing_expectations[i - 1] in final_answers:
                    print("PASSED")
                    passed += 1
                else:
                    print("Expected " + testing_expectations[i - 1] + " but got: " + str(final_answers))
            except:
                print("Something wrong occured")

        print("-----FINAL ACCURACY-------")
        print("\t" + str(passed / 10))