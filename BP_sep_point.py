import numpy as np
from scipy.interpolate import CubicSpline
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

index = 0

# test
ptt_filename = dir_list[index]  # num from 0~5 (6 participants)

# data loading
ptt = pd.read_csv(ptt_filename)
ptt_df = pd.DataFrame(ptt)
ptt_list = ptt_df.values.tolist()
ptt_array = np.array(ptt_list)

PTT = ptt_array[:, 2].astype(float)
HR = ptt_array[:, 1].astype(float)
SBP = ptt_array[:, 3].astype(float)
DBP = ptt_array[:, 4].astype(float)


# for i in range(len(dir_list)):
#     if i == 0:
#         ptt_filename = dir_list[i]
#         print(ptt_filename)
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



sep_point = []

num = 0
for n in range(len(SBP)):
    if SBP[n] != num:
        sep_point.append(n)
        num = SBP[n]

sep_SBP = [SBP[n] for n in sep_point]
print(len(sep_point))
print(sep_point)
print(sep_SBP)


x_fine = np.linspace(0, len(SBP), len(SBP))

f = CubicSpline(sep_point, sep_SBP, bc_type='natural')
plt.plot(sep_point, sep_SBP, 'ko')
plt.plot(x_fine, f(x_fine), '-', alpha=0.8)
print(x_fine)
print(f(x_fine))
plt.show()


save_path = os.path.join(root, 'data', f'{index}')

if not os.path.isdir(save_path):
    os.mkdir(save_path)


PTT_discrete = [PTT[n] for n in sep_point]
HR_discrete = [HR[n] for n in sep_point]
SBP_discrete = [SBP[n] for n in sep_point]



os.chdir(save_path)
np.save(f'PTT_continuous_{index}.npy', PTT)
np.save(f'HR_continuous_{index}.npy', HR)
np.save(f'SBP_continuous_{index}.npy', SBP)
np.save(f'waveform data_{index}.npy', f(x_fine))
np.save(f'BP_sep_point_{index}.npy', sep_point)
np.save(f'SBP_discrete_{index}.npy', SBP_discrete)
np.save(f'PTT_discrete_{index}.npy', PTT_discrete)
np.save(f'HR_discrete_{index}.npy', HR_discrete)