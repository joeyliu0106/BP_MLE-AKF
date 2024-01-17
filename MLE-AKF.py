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

# data reading
root = os.getcwd()
file_path = os.path.join(root, 'data', 'whole data')
dir_list = os.listdir(file_path)
os.chdir(file_path)

# #####   data loading   #####
index = 1
# test
ptt_filename = dir_list[index]  # num from 0~5 (6 participants)

# data loading
ptt = pd.read_csv(ptt_filename)
ptt_df = pd.DataFrame(ptt)
ptt_list = ptt_df.values.tolist()
ptt_array = np.array(ptt_list)

PTT = ptt_array[:, 2].astype(float)
print(PTT**-1)
HR = ptt_array[:, 1].astype(float)
old_SBP = ptt_array[:, 3].astype(float)
DBP = ptt_array[:, 4].astype(float)


# for i in range(len(dir_list)):
#     if i == 0:
#         ptt_filename = dir_list[i]
#
#         # data loading
#         ptt = pd.read_csv(ptt_filename)
#         ptt_df = pd.DataFrame(ptt)
#         ptt_list = ptt_df.values.tolist()
#         ptt_array = np.array(ptt_list)
#
#         PTT = ptt_array[:, 2].astype(float)
#         HR = ptt_array[:, 1].astype(float)
#         SBP = ptt_array[:, 3].astype(float)
#         DBP = ptt_array[:, 4].astype(float)
#
#     elif i > 0:
#         ptt_filename = dir_list[i]
#
#         # data loading
#         ptt = pd.read_csv(ptt_filename)
#         ptt_df = pd.DataFrame(ptt)
#         ptt_list = ptt_df.values.tolist()
#         ptt_array = np.array(ptt_list)
#
#         PTT = np.concatenate((PTT, ptt_array[:, 2].astype(float)), axis=0)
#         HR = np.concatenate((HR, ptt_array[:, 1].astype(float)), axis=0)
#         SBP = np.concatenate((SBP, ptt_array[:, 3].astype(float)), axis=0)
#         DBP = np.concatenate((DBP, ptt_array[:, 4].astype(float)), axis=0)
#
# print(PTT.shape)
# print(HR.shape)
# print(SBP.shape)
# print(DBP.shape)


os.chdir(root)

SBP = np.load(f'waveform data_{index}.npy')
sep_point = np.load(f'BP_sep_point_{index}.npy')
sep_SBP = [SBP[n] for n in sep_point]



counter = 0
calibration_period = 100    # calibration interval
result_list = []
old_result_list = []

for i in range(len(SBP)):
    print(counter + 1)
    X1 = np.zeros((3, 1), dtype=float)
    X1[0, 0] = 1
    X1[1, 0] = PTT[i]*1000
    print(PTT[i]*1000)
    X1[2, 0] = HR[i]
    print(HR[i])

    if counter % calibration_period == 0:
        coeff, sig_c = MLE_AKF_predict(PTT*1000, HR, SBP, 20, counter)  # a: how many samples for each MLE calculation

        if len(coeff) != 0:
            if counter == 0:
                sig_r = np.dot(X1, X1.T)
                C = np.copy(coeff)
            else:
                sig_r = inv(inv(sig_r + sig_c) + np.dot(X1, X1.T))
                coeff = MLE_AKF_correction(PTT*1000, HR, SBP, coeff, sig_r, counter)
                coeff_back_up = np.copy(coeff)
        else:
            coeff = coeff_back_up

        print('Coefficient: ', coeff)


    result = np.dot(coeff.T, X1)
    old_result = np.dot(C.T, X1)

    result_list.append(result)
    old_result_list.append(old_result)
    print('golden: ', SBP[i])
    print('estimated: ', result)

    counter += 1

result_list = np.array(result_list)
result_list = result_list.reshape(len(result_list))
old_result_list = np.array(old_result_list)
old_result_list = old_result_list.reshape(len(old_result_list))

x = list(range(0, len(SBP)))
x = np.array(x)
plt.figure()
plt.xlabel('Samples')
plt.ylabel('SBP value')
plt.title('Estimating Result')
plt.plot(x, result_list, label='estimated')
plt.plot(x, old_result_list, label='estimated_old')
plt.plot(x, SBP, label='SBP')
plt.plot(x, old_SBP, label='old_SBP')
plt.plot(sep_point, sep_SBP, 'ko')
plt.legend()
plt.show()



plt.figure()
plt.xlabel('PTT')
plt.ylabel('SBP')
plt.scatter(PTT*1000, SBP, s=1)
plt.show()

print('#####################################################')
mse = mean_squared_error(SBP, result_list)
rmse = sqrt(mean_squared_error(SBP, result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)
print('correlation matrix:\n', np.corrcoef(SBP, result_list))