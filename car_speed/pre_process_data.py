import os
import numpy as np
import cv2
from progress.bar import Bar

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_TRAIN_VID = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'train.mp4'))
PATH_TRAIN_SPEED = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'train.txt'))
PATH_TEST_VID = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'test.mp4'))

def train_data(step = 200, callback = None):
    cap = cv2.VideoCapture(PATH_TRAIN_VID)
    success, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    count = 1
    totalCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    bar = Bar('Generating training data', max = totalCount)

    data = []
    output = train_output()
    while success:
        success, frame2 = cap.read()
        if not np.any(frame2):
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        formatted = []
        for x in bgr:
            for y in x:
                formatted.extend(y)

        data.append(formatted)
        if len(data) > step:
            callback(data, output[count:count + step + 1])
            data = []

        prvs = next
        count += 1
        bar.next()

    if len(data) > 0:
        callback(data, output[len(output) - len(data):len(output)])

    bar.finish()

def train_output():
    with open(PATH_TRAIN_SPEED, 'r') as file:
        speed_data = file.read().split('\n')

    return speed_data

def test_data():
    cap = cv2.VideoCapture(PATH_TEST_VID)
    success, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    count = 1 # first and last frame are unable to be processed
    totalCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 2
    bar = Bar('Generating testing data', max = totalCount)

    data = []
    while success:
        success, frame2 = cap.read()
        if not np.any(frame2):
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        formatted = []
        for x in bgr:
            for y in x:
                formatted.extend(y)

        data.append(formatted)

        if len(data) > step:
            callback(data)
            data = []

        prvs = next
        count += 1
        bar.next()

    if len(data) > 0:
        callback(data)

    bar.finish()
