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

# test
ptt_filename = dir_list[0]  # num from 0~5 (6 participants)

# data loading
ptt = pd.read_csv(ptt_filename)
ptt_df = pd.DataFrame(ptt)
ptt_list = ptt_df.values.tolist()
ptt_array = np.array(ptt_list)

PTT = ptt_array[:, 2].astype(float)
HR = ptt_array[:, 1].astype(float)
SBP = ptt_array[:, 3].astype(float)
DBP = ptt_array[:, 4].astype(float)




counter = 0
calibration_period = 100    # calibration interval
result_list = []
old_result_list = []

for i in range(len(SBP)):
    print(counter + 1)
    X1 = np.zeros((3, 1), dtype=float)
    X1[0, 0] = 1
    X1[1, 0] = PTT[i]
    X1[2, 0] = HR[i]

    if counter == 0:
        coeff, sig_c = MLE_AKF_predict(PTT*100, HR, SBP, 10, counter)       # a: how many samples for each MLE calculation
        sig_r0 = np.dot(X1, X1.T)
        sig_r = np.copy(sig_r0)
        C = np.copy(coeff)

    elif counter != 0 and counter % calibration_period == 0:
        sig_r = inv(inv(sig_r + sig_c) + np.dot(X1, X1.T))
        coeff = MLE_AKF_correction(PTT, HR, SBP, coeff, sig_r, counter)



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
plt.plot(x, result_list, 'black', label='estimated')
plt.plot(x, SBP, 'blue', label='SBP')
plt.legend()
plt.show()

print('#####################################################')
mse = mean_squared_error(SBP, result_list)
rmse = sqrt(mean_squared_error(SBP, result_list))
print('MSE: ', mse)
print('RMSE: ', rmse)