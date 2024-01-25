import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from functions import *
import pandas as pd
from statistics import mean
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)
np.set_printoptions(suppress=True)

index = 0   # file index(0~5, 'all')

######   data reading   ######
root = os.getcwd()
file_path = os.path.join(root, 'data', f'{index}')
os.chdir(file_path)


# common usage
wave = np.load(f'waveform data_{index}.npy')
sep_point = np.load(f'BP_sep_point_{index}.npy')
# # discrete
# SBP = np.load(f'SBP_discrete_{index}.npy')
# HR = np.load(f'HR_discrete_{index}.npy')
# PTT = np.load(f'PTT_discrete_{index}.npy')
# continuous
SBP = np.load(f'waveform data_{index}.npy')
HR = np.load(f'HR_continuous_{index}.npy')
PTT = np.load(f'PTT_continuous_{index}.npy')

print('number of samples: ', len(SBP))

X = np.zeros((3, len(SBP)), dtype=float)
X[0] = [1 for i in range(len(SBP))]
X[1] = [PTT[i] for i in range(len(SBP))]
X[2] = [HR[i] for i in range(len(SBP))]
print(X[:, 0].reshape(3, 1))


######   parameters   ######
akf_counter = 0
mle_end_flag = 0
akf_exe_flag = 0
calibration_period = 3    # calibration interval
registration_num = 3      # calibration number for MLE

A = np.zeros((3, 3), dtype=float)
a = np.zeros((3, 1), dtype=float)
sig_c = 0

result_list = []
result_index_list = []
update_index = []
error = []

######   MLE-AKF   ######
for n in range(len(sep_point)):
    print('######   sample number: ', n + 1, '\n')
    input = X[:, sep_point[n]].reshape((3, 1))

    # coefficient calculation
    if n < registration_num:
        # MLE
        coeff, A, a = MLE(input, SBP[sep_point[n]], A, a)

        if n == registration_num - 1:
            mle_end_flag = 1
            print('MLE ended')

    # decide if AKF should work or not
    if n != 0 and (mle_end_flag == 1 or ((n + 1) - registration_num) % calibration_period == 0):
        akf_exe_flag = 1
        mle_end_flag = 0
        print('AKF activated')

    # executing AKF
    if akf_exe_flag == 1:
        # previous coefficient remembering
        coeff_old = np.copy(coeff)

        if akf_counter == 0:
            sig_r = np.dot(X[:, 0].reshape((3, 1)), X[:, 0].reshape((3, 1)).T)
        sig_r = pinv(pinv(sig_r + sig_c) + np.dot(input, input.T))

        coeff = AKF(coeff_old, input, sig_r, SBP[sep_point[n]])
        print('coefficient updated!!!')
        print('previous coefficient:\n', coeff_old)
        print('coefficient:\n', coeff)
        # sig_c update
        sig_c = coeff_cov(coeff_old, coeff)
        print('sig_c update!!!')
        print('sig_c:\n', sig_c)
        akf_counter = akf_counter + 1
        akf_exe_flag = 0

        update_index.append(sep_point[n])   # optional

    if akf_counter > 0:
        if n < len(sep_point) - 1:
            for i in range(sep_point[n], sep_point[n + 1]):
                input = X[:, i].reshape((3, 1))

                result = np.dot(coeff.T, input)
                result_list.append(result)
                result_index_list.append(i)

                print('golden: ', SBP[i])
                print('estimated: ', result)



result_list = np.array(result_list)
result_list = np.reshape(result_list, len(result_index_list))

######   averaging   ######
avg_num = 60
avg_SBP_list = []
avg_result_list = []
avg_result_index = []

for i in range(len(result_list)):
    if i % avg_num == 0:
        avg_result = mean(result_list[i:(i + avg_num - 1)])
        avg_result_list.append(avg_result)
        avg_result_index.append(result_index_list[i])

        avg_SBP = mean(SBP[i:(i + avg_num - 1)])
        avg_SBP_list.append(avg_SBP)


######   visualization   ######
# x = list(range(0, len(result_list)))
x = [n for n in range(sep_point[calibration_period], len(result_list)+sep_point[calibration_period])]
x_SBP = list(range(0, len(SBP)))
SBP_update = [SBP[n] for n in update_index]
print('sep point index: ', sep_point)
print('update index: ', update_index)

plt.figure()
plt.xlabel('Samples')
plt.ylabel('SBP value')
plt.title('Estimating Result')
plt.plot(x, result_list, label='whole estimated')
plt.plot(x_SBP, SBP, label='golden')
# plt.plot(avg_result_index, avg_result_list, label='avg estimated')
# plt.plot(avg_result_index, avg_result_list, 'ko')
# plt.plot(avg_result_index, avg_SBP_list, label='avg golden')
# plt.plot(avg_result_index, avg_SBP_list, 'ko')
plt.plot([n for n in sep_point], [SBP[n] for n in sep_point], 'ko')
plt.plot(update_index, SBP_update, 'ro')
plt.legend()
plt.show()


######   criteria print out   ######
print('#####################################################')
mse = mean_squared_error(avg_SBP_list, avg_result_list)
rmse = sqrt(mean_squared_error(avg_SBP_list, avg_result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)
print('correlation matrix:\n', np.corrcoef(avg_SBP_list, avg_result_list))