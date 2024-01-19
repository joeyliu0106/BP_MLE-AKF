import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
import pandas as pd
import sys
import os
from scipy import stats
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


def MLE(X, BP, A, a):
    A = np.dot(X, X.T) + A
    a = np.dot(X, BP) + a

    A_inv = pinv(A)
    C = np.dot(A_inv, a)

    return C, A, a


def AKF(coeff, X, sig_R, BP):
    coeff_new = coeff + np.dot(sig_R, X) * (BP - np.dot(coeff.T, X))

    return coeff_new


def cov_coeff(coeff):
    sig_C = np.array([np.cov([coeff[0, 0], coeff[0, 0]]), np.cov([coeff[0, 0], coeff[1, 0]]), np.cov([coeff[0, 0], coeff[2, 0]]),
                      np.cov([coeff[1, 0], coeff[0, 0]]), np.cov([coeff[1, 0], coeff[1, 0]]), np.cov([coeff[1, 0], coeff[2, 0]]),
                      np.cov([coeff[2, 0], coeff[0, 0]]), np.cov([coeff[2, 0], coeff[1, 0]]), np.cov([coeff[2, 0], coeff[2, 0]])]).reshape(3, 3)
    return sig_C




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



def var_cal(err):
    zero_mean = stats.zscore(err)
    var = np.var(zero_mean)
    return var