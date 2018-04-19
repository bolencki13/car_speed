import os
import numpy as np
import cv2
import pickle
from progress.bar import Bar

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN_VID = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'train.mp4'))
PATH_TRAIN_SPEED = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'train.txt'))
PATH_DATA = os.path.abspath(os.path.join(PATH, os.pardir, 'frame_data'))

def convert_video_to_frames(videoPath, speedPath = None, dumpPath = None):
    if speedPath:
        with open(PATH_TRAIN_SPEED, 'r') as file:
            speed_data = file.read().split('\n')

    cap = cv2.VideoCapture(videoPath)
    success, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    formatted_data = []
    count = 1
    totalCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar('Converting video to useable data format', max = totalCount)

    while success:
        success, frame2 = cap.read()
        if not np.any(frame2):
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        prvs = next

        if speedPath:
            formatted_data.append([hsv, speed_data[count]])
        else:
            formatted_data.append([hsv])

        bar.next()
        count += 1

    bar.finish()

    if dumpPath:
        with open(dumpPath, 'wb') as file:
            pickle.dump(formatted_data, file)

        print("Saved data to file:", dumpPath)

    return formatted_data


if __name__ == '__main__':
    path_out = os.path.abspath(os.path.join(PATH_DATA, 'train.pickle'))
    convert_video_to_frames(PATH_TRAIN_VID, PATH_TRAIN_SPEED, path_out)
