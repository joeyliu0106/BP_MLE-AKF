import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
import pandas as pd
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


def data_reading(file_path):
    dir_list = os.listdir(file_path)
    os.chdir(file_path)

    for i in range(len(dir_list)):
        if i == 0:
            ptt_filename = dir_list[i]

            # data loading
            ptt = pd.read_csv(ptt_filename)
            ptt_df = pd.DataFrame(ptt)
            ptt_list = ptt_df.values.tolist()
            ptt_array = np.array(ptt_list)

            PTT = ptt_array[:, 1].astype(float)
            HR = ptt_array[:, 3].astype(float)
            SBP = ptt_array[:, 4].astype(float)
            DBP = ptt_array[:, 5].astype(float)

        elif i > 0:
            ptt_filename = dir_list[i]

            # data loading
            ptt = pd.read_csv(ptt_filename)
            ptt_df = pd.DataFrame(ptt)
            ptt_list = ptt_df.values.tolist()
            ptt_array = np.array(ptt_list)

            PTT = np.concatenate((PTT, ptt_array[:, 1].astype(float)), axis=0)
            HR = np.concatenate((HR, ptt_array[:, 2].astype(float)), axis=0)
            SBP = np.concatenate((SBP, ptt_array[:, 3].astype(float)), axis=0)
            DBP = np.concatenate((DBP, ptt_array[:, 4].astype(float)), axis=0)

    return PTT, HR, SBP, DBP


def MLE_AKF_predict(PTT, HR, BP, a, counter):
    # a: numbers of calibration measurements
    # predict
    if counter + a <= len(PTT):
        for n in range(a):
            X = np.zeros((3, 1), dtype=float)
            X[0, 0] = 1
            X[1, 0] = PTT[counter + n]
            X[2, 0] = HR[counter + n]

            if n == 0:
                A = np.dot(X, X.T)
                a = np.dot(X, BP[counter + n])
            elif n > 0:
                A = A + np.dot(X, X.T)
                a = a + np.dot(X, BP[counter + n])

        A_inv = pinv(A)
        C = np.dot(A_inv, a)

        sig_C = np.array([np.cov([C[0, 0], C[0, 0]]), np.cov([C[0, 0], C[1, 0]]), np.cov([C[0, 0], C[2, 0]]),
                          np.cov([C[1, 0], C[0, 0]]), np.cov([C[1, 0], C[1, 0]]), np.cov([C[1, 0], C[2, 0]]),
                          np.cov([C[2, 0], C[0, 0]]), np.cov([C[2, 0], C[1, 0]]), np.cov([C[2, 0], C[2, 0]])]).reshape(3, 3)

        return C, sig_C
    else:
        return [], []


def MLE_AKF_correction(PTT, HR, BP, C, sig_r, counter):
    # correction
    X = np.zeros((3, 1), dtype=float)
    X[0, 0] = 1
    X[1, 0] = PTT[counter]
    X[2, 0] = HR[counter]

    golden = BP[counter]
    predict = np.dot(C.T, X)

    C_new = C + np.dot(sig_r, X) * (golden - predict)

    return C_new