import os
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pre_process_data
import pickle

PATH = os.path.dirname(os.path.abspath(__file__))
PATH_CLASSIFIER = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'classifier.pickle'))
PATH_TEST_DATA = os.path.abspath(os.path.join(PATH, os.pardir, 'data', 'text.txt'))

def train_batch(x, y):
    if os.path.exists(PATH_CLASSIFIER) == False:
        clf = LinearRegression(n_jobs=-1) #svm.SVR(kernel='poly')
    else:
        with open(PATH_CLASSIFIER, 'rb') as file:
            clf = pickle.load(file)

    clf.fit(x, y)
    with open(PATH_CLASSIFIER, 'wb') as file:
        pickle.dump(clf, file)

def test_batch(x):
    with open(PATH_CLASSIFIER, 'rb') as file:
        clf = pickle.load(file)

    y_test = clf.predict(x)
    with open(PATH_TEST_DATA, "a") as file:
        for y in y_test:
            output = str(y) + "\n"
            file.write(output)

def callback_train(x, y):
    print('Training new data batch')
    train_batch(x, y)

def callback_test(x):
    print('Testing new data batch')
    test_batch(x)

if __name__ == '__main__':
    pre_process_data.train_data(callback = callback_train)

    print('Training done')

    try:
        os.remove(PATH_TEST_DATA)
    except OSError:
        pass

    pre_process_data.test_data(callback = callback_test)
    print('Testing done')
