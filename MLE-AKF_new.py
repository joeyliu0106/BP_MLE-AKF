# import matplotlib.pyplot as plt
# import numpy as np
# from numpy.linalg import *
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from functions import *
# import pandas as pd
# import sys
# import os
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(linewidth=1000)
#
# index = 'all'
#
# # data reading
# root = os.getcwd()
# file_path = os.path.join(root, 'data', f'{index}')
# os.chdir(file_path)
#
#
# # common usage
# wave = np.load(f'waveform data_{index}.npy')
# sep_point = np.load(f'BP_sep_point_{index}.npy')
# # discrete
# SBP = np.load(f'SBP_discrete_{index}.npy')
# HR = np.load(f'HR_discrete_{index}.npy')
# PTT = np.load(f'PTT_discrete_{index}.npy')
# # # continuous
# # SBP = np.load(f'waveform data_{index}.npy')
# # HR = np.load(f'HR_continuous_{index}.npy')
# # PTT = np.load(f'PTT_continuous_{index}.npy')
#
# print('number of samples: ', len(SBP))
#
# # parameters
# counter = 0
# calibration_period = 100    # calibration interval
# cali_num = 5               # calibration number for MLE
#
# A = np.zeros((3, 3), dtype=float)
# a = np.zeros((3, 1), dtype=float)
#
# result_list = []
# result_list_old = []
# error = []
#
# # MLE-AKF
# for n in range(len(PTT)):
#     print('sample number: ', n + 1)
#
#     if n % calibration_period == 0:
#         # perform MLE
#         if n == 0:
#             for i in range(cali_num):
#                 X = np.zeros((3, 1), dtype=float)
#                 X[0, 0] = 1
#                 X[1, 0] = PTT[n + i]
#                 X[2, 0] = HR[n + i]
#
#                 coeff, A, a = MLE(X, SBP[n + i], A, a)
#                 coeff_old = np.copy(coeff)
#                 sig_C = cov_coeff(coeff)
#                 print('coefficient updated by MLE')
#         # perform AKF
#         else:
#             X = np.zeros((3, 1), dtype=float)
#             X[0, 0] = 1
#             X[1, 0] = PTT[n]
#             X[2, 0] = HR[n]
#             golden = SBP[n]
#
#             sig_R = np.dot(X, X.T)
#             sig_R = inv(inv(sig_R + sig_C) + np.dot(X, X.T))
#             coeff = AKF(coeff, X, sig_R, golden)
#
#             sig_C = cov_coeff(coeff)
#             print('coefficient updated by AKF')
#
#     X = np.zeros((3, 1), dtype=float)
#     X[0, 0] = 1
#     X[1, 0] = PTT[n]
#     X[2, 0] = HR[n]
#
#     result = np.dot(coeff.T, X)
#     result_list.append(result)
#     result_old = np.dot(coeff_old.T, X)
#     result_list_old.append(result_old)
#
#     print('golden: ', SBP[n])
#     print('estimated: ', result)
#     print('estimated_old: ', result_old)
#
# result_list = np.array(result_list)
# result_list = np.reshape(result_list, len(SBP))
# result_list_old = np.array(result_list_old)
# result_list_old = np.reshape(result_list_old, len(SBP))
#
#
# # visualization
# x = list(range(0, len(SBP)))
#
# plt.figure()
# plt.xlabel('Samples')
# plt.ylabel('SBP value')
# plt.title('Estimating Result')
# plt.plot(x, result_list, label='estimated')
# # plt.plot(x, result_list, 'ko')
# # plt.plot(x, result_list_old, label='estimated_old')
# # plt.plot(x, result_list_old, 'ko')
# plt.plot(x, SBP, label='golden')
# # plt.plot(x, SBP, 'ko')
# plt.legend()
# plt.show()
#
#
#
# print('#####################################################')
# mse = mean_squared_error(SBP, result_list)
# rmse = sqrt(mean_squared_error(SBP, result_list))
# print('MSE: ', mse)
# print('RMSE: ', rmse)
# print('correlation matrix:\n', np.corrcoef(SBP, result_list))
#
# print(np.cov([1, 2, 8], [4, 6, 6]))






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

index = 5   # file index(0~5, 'all')

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
calibration_period = 70    # calibration interval
registration_num = 30       # calibration number for MLE

A = np.zeros((3, 3), dtype=float)
a = np.zeros((3, 1), dtype=float)
sig_c = 0

result_list = []
result_index_list = []
error = []

######   MLE-AKF   ######
for n in range(len(SBP)):
    print('sample number: ', n + 1)
    input = X[:, n].reshape((3, 1))

    # coefficient calculation
    if n < registration_num:
        # MLE
        coeff, A, a = MLE(input, SBP[n], A, a)

        if n == registration_num - 1:
            mle_end_flag = 1
            print('MLE ended')

    # decide if AKF should work or not
    if mle_end_flag == 1 or ((n + 1) - registration_num) % calibration_period == 0:
        akf_exe_flag = 1
        mle_end_flag = 0
        print('AKF activated')

    # executing AKF
    if akf_exe_flag == 1:
        # previous coefficient remembering
        coeff_before = np.copy(coeff)

        if akf_counter == 0:
            sig_r = np.dot(X[:, 0].reshape((3, 1)), X[:, 0].reshape((3, 1)).T)
        sig_r = pinv(pinv(sig_r + sig_c) + np.dot(input, input.T))

        coeff = AKF(coeff, input, sig_r, SBP[n])
        # sig_c update
        sig_c = coeff_cov(coeff_before, coeff)
        akf_counter = akf_counter + 1
        akf_exe_flag = 0

    if akf_counter > 0:
        result = np.dot(coeff.T, input)
        result_list.append(result)
        result_index_list.append(n)

        print('golden: ', SBP[n])
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
x = list(range(0, len(result_list)))
x_SBP = list(range(0, len(SBP)))


plt.figure()
plt.xlabel('Samples')
plt.ylabel('SBP value')
plt.title('Estimating Result')
# plt.plot(x, result_list, label='whole estimated')
# plt.plot(x_SBP, SBP, label='golden')
plt.plot(avg_result_index, avg_result_list, label='avg estimated')
plt.plot(avg_result_index, avg_result_list, 'ko')
plt.plot(avg_result_index, avg_SBP_list, label='avg golden')
plt.plot(avg_result_index, avg_SBP_list, 'ko')
plt.legend()
plt.show()


######   criteria print out   ######
print('#####################################################')
mse = mean_squared_error(avg_SBP_list, avg_result_list)
rmse = sqrt(mean_squared_error(avg_SBP_list, avg_result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)
print('correlation matrix:\n', np.corrcoef(avg_SBP_list, avg_result_list))