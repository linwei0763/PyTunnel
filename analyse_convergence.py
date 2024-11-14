import fnmatch
import numpy as np
import os
import pandas as pd


if __name__ == '__main__':

    '''config'''
    path_o = 'result'
    stations = [0]
    r_all = [2.75]
    angles = [202.5, 112.5, 180, 157.5, 135, 90, 45]
    '''config'''
    
    convergence = []
    for i in range(len(stations)):
        station = stations[i]
        files = fnmatch.filter(os.listdir(path_o), str(station) + '-*-ellipse-polynomial.xlsx')
        r = r_all[i]
        for file in files:
            convergence.append([int(file.split('-')[0]), int(file.split('-')[1]), int(file.split('-')[2])])
            data = pd.read_excel(os.path.join(path_o, file), header=None, names=None)
            data = np.asarray(data)
            data = data[:, 0:2]
            polar = np.arctan2(data[:, 1], data[:, 0]) / np.pi * 180
            for angle in angles:
                index = (np.abs(polar - angle)).argmin()
                c = np.linalg.norm(data[index, :]) - r
                index = (np.abs(polar - (angle - 180))).argmin()
                c += np.linalg.norm(data[index, :]) - r
                convergence[-1].append(c*1000)
    
    convergence = pd.DataFrame(convergence)
    convergence.to_excel(os.path.join(path_o, 'convergence.xlsx'), header=False, index=False)