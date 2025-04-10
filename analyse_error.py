import fnmatch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score


def customise_histogram(data, path_output):
    
    '''------'''
    config = {
        'axes.unicode_minus': True,
        'font.sans-serif': ['Times New Roman',],
        'font.size': 7,
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic'
    }
    rcParams.update(config)
    
    plt.figure(figsize=(3.5 / 2.54, 4 / 2.54), dpi=600)
    
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    '''------'''
    
    '''------'''
    plt.xlim(-10, 10)
    plt.ylim(0, 0.2)
    
    plt.xlabel('Fitting error (mm)')
    plt.ylabel('Frequency')
    
    weights = np.ones_like(data) / float(len(data))
    
    plt.hist(data, bins=45, range=(-10, 10), weights=weights, color='#a9a9a9')
    
    mu = np.mean(data)
    sigma = np.std(data)
    plt.text(0, 0.18, r'$\it{\mu}$$\rm=%.3f$  $\it{\sigma}$$\rm=%.3f$'%(mu, sigma), ha='center')
    '''------'''
    
    '''------'''
    plt.savefig(path_output + '.tiff', dpi=600, bbox_inches='tight')
    plt.savefig(path_output + '.eps', dpi=600, bbox_inches='tight')
    plt.close()
    return mu, sigma
    '''------'''


if __name__ == '__main__':
    
    '''---config---'''
    path_o = 'result'
    
    name_error = ['ellipse', 'seg_circle', 'tff', 'ftf', 'ftt']
    index_error = [-11, -9, -7, -5 ,-3]
    index_d0 = -13
    index_joint = -1
    
    name_tunnel = [[0], [1, 2], [3], [4, 5]]
    '''---config---'''
    
    '''all'''
    files = fnmatch.filter(os.listdir(path_o), '*.txt')
    errors = []
    for file in files:
        pc = np.loadtxt(os.path.join(path_o, file), delimiter=' ')
        e = pc[:, index_error[:]]
        d0 = pc[:, index_d0].reshape(-1, 1)
        joint = pc[:, index_joint].reshape(-1, 1)
        tunnel_no = np.zeros((e.shape[0], 1))
        tunnel_no[:, 0] = int(file.split('-')[0])
        e = np.hstack((e, d0, joint, tunnel_no))
        errors.append(e)
        # if int(file.split('-')[0]) in [1, 2]:
        #     if np.max(e[:, 2]) >= 1:
        #         print(file)
        #         print(np.max(e[:, 2]))
        #     if np.min(e[:, 2]) <= -1:
        #         print(file)
        #         print(np.min(e[:, 2]))
    errors = np.vstack((errors[:]))
    errors[:, :-2] = errors[:, :-2] * 1000
    statistics = []
    
    ratio_all = []
    
    for i in range(len(name_tunnel)):
        statistics.append([])
        d0 = errors[np.where(np.isin(errors[:, -1], name_tunnel[i]))[0], -3]
        for j in range(len(index_error)):
            error = errors[np.where(np.isin(errors[:, -1], name_tunnel[i]))[0], j]
            d = d0 - error
            error_in = error[np.where((error <= 10) & (error >= -10))[0]]
            ratio = (error.shape[0] - error_in.shape[0]) / error.shape[0]
            ratio_all.append(ratio)
            path_output = os.path.join(path_o, str(name_tunnel[i][0]) + '-' + name_error[j])
            mu, sigma = customise_histogram(error_in, path_output)
            mae = np.mean(np.abs(error))
            r2 = r2_score(d0, d)
            statistics[-1].append(mu)
            statistics[-1].append(sigma)
            statistics[-1].append(mae)
            statistics[-1].append(r2)
    
    ratio_all = np.asarray(ratio_all)
    print(np.max(ratio_all))
    
    statistics = np.asarray(statistics)
    statistics = pd.DataFrame(statistics)
    statistics.to_excel(os.path.join(path_o, 'statistics.xlsx'), header=False, index=False)
    '''all'''
    
    '''joint'''
    errors = errors[np.where(errors[:, -2] != 0)[0], :]
    statistics = []
    
    for i in range(len(name_tunnel)):
        statistics.append([])
        d0 = errors[np.where(np.isin(errors[:, -1], name_tunnel[i]))[0], -3]
        for j in range(len(index_error)):
            error = errors[np.where(np.isin(errors[:, -1], name_tunnel[i]))[0], j]
            d = d0 - error
            error_in = error[np.where((error <= 10) & (error >= -10))[0]]
            path_output = os.path.join(path_o, 'j' + str(name_tunnel[i][0]) + '-' + name_error[j])
            mu, sigma = customise_histogram(error_in, path_output)
            mae = np.mean(np.abs(error))
            r2 = r2_score(d0, d)
            statistics[-1].append(mu)
            statistics[-1].append(sigma)
            statistics[-1].append(mae)
            statistics[-1].append(r2)
    
    statistics = np.asarray(statistics)
    statistics = pd.DataFrame(statistics)
    statistics.to_excel(os.path.join(path_o, 'statistics_joint.xlsx'), header=False, index=False)
    '''joint'''