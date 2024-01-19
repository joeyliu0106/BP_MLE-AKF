import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from functions import *
import pandas as pd
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)

index = 0

# data reading
root = os.getcwd()
file_path = os.path.join(root, 'data', f'{index}')
os.chdir(file_path)

SBP = np.load(f'SBP_discrete_{index}.npy')
HR = np.load(f'HR_discrete_{index}.npy')
PTT = np.load(f'PTT_discrete_{index}.npy')
wave = np.load(f'waveform data_{index}.npy')
sep_point = np.load(f'BP_sep_point_{index}.npy')

print('number of samples: ', len(sep_point))

# parameters
counter = 0
calibration_period = 3    # calibration interval
cali_num = 1               # calibration number for MLE

A = np.zeros((3, 3), dtype=float)
a = np.zeros((3, 1), dtype=float)

result_list = []
result_list_old = []
error = []

# MLE-AKF
for n in range(len(sep_point)):
    print('sample number: ', n + 1)

    if n % calibration_period == 0:
        # perform MLE
        if n == 0:
            for i in range(cali_num):
                X = np.zeros((3, 1), dtype=float)
                X[0, 0] = 1
                X[1, 0] = PTT[n + i]
                X[2, 0] = HR[n + i]

                coeff, A, a = MLE(X, SBP[n + i], A, a)
                coeff_old = np.copy(coeff)
                sig_C = cov_coeff(coeff)
                print('coefficient updated by MLE')
        # perform AKF
        else:
            X = np.zeros((3, 1), dtype=float)
            X[0, 0] = 1
            X[1, 0] = PTT[n]
            X[2, 0] = HR[n]
            golden = SBP[n]

            sig_R = np.dot(X, X.T)
            sig_R = inv(inv(sig_R + sig_C) + np.dot(X, X.T))
            coeff = AKF(coeff, X, sig_R, golden)

            sig_C = cov_coeff(coeff)
            print('coefficient updated by AKF')

    X = np.zeros((3, 1), dtype=float)
    X[0, 0] = 1
    X[1, 0] = PTT[n]
    X[2, 0] = HR[n]

    result = np.dot(coeff.T, X)
    result_list.append(result)
    result_old = np.dot(coeff_old.T, X)
    result_list_old.append(result_old)

    print('golden: ', SBP[n])
    print('estimated: ', result)
    print('estimated_old: ', result_old)

result_list = np.array(result_list)
result_list = np.reshape(result_list, len(sep_point))
result_list_old = np.array(result_list_old)
result_list_old = np.reshape(result_list_old, len(sep_point))


# visualization
x = list(range(0, len(sep_point)))

plt.figure()
plt.xlabel('Samples')
plt.ylabel('SBP value')
plt.title('Estimating Result')
plt.plot(x, result_list, label='estimated')
plt.plot(x, result_list, 'ko')
plt.plot(x, result_list_old, label='estimated_old')
plt.plot(x, result_list_old, 'ko')
plt.plot(x, SBP, label='golden')
plt.plot(x, SBP, 'ko')
plt.legend()
plt.show()



print('#####################################################')
mse = mean_squared_error(SBP, result_list)
rmse = sqrt(mean_squared_error(SBP, result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)
print('correlation matrix:\n', np.corrcoef(SBP, result_list))