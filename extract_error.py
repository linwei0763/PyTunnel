import fnmatch
import numpy as np
import os
import pandas as pd


'''---config---'''
path_o = 'result'
index_error = [-10, -8, -6, -4 ,-2]
'''---config---'''


files = fnmatch.filter(os.listdir(path_o), '*.txt')

errors = []

for file in files:
    pc = np.loadtxt(os.path.join(path_o, file), delimiter=' ')
    e = pc[:, index_error[:]]
    tunnel_no = np.zeros((e.shape[0], 1))
    tunnel_no[:, 0] = int(file.split('-')[0])
    e = np.hstack((e, tunnel_no))
    errors.append(e)

errors = np.vstack((errors[:]))
errors = pd.DataFrame(errors)
errors.to_csv(os.path.join(path_o, 'error.csv'), header=False, index=False)