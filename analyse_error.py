import fnmatch
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


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
    
    plt.xlabel('Fitting error / mm')
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
    index_joint = -1
    index_error = [-10, -8, -6, -4 ,-2, index_joint]
    
    name_tunnel = [[0], [1, 2], [3], [4, 5]]
    specific_index = [0, 1, 2, 3, 4]
    '''---config---'''
    
    '''all'''
    files = fnmatch.filter(os.listdir(path_o), '*.txt')
    errors = []
    for file in files:
        pc = np.loadtxt(os.path.join(path_o, file), delimiter=' ')
        e = pc[:, index_error[:]]
        tunnel_no = np.zeros((e.shape[0], 1))
        tunnel_no[:, 0] = int(file.split('-')[0])
        e = np.hstack((e, tunnel_no))
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
    
    # ratio_all = []
    
    for i in range(len(name_tunnel)):
        statistics.append([])
        for j in range(len(specific_index)):
            error = errors[np.where(np.isin(errors[:, -1], name_tunnel[i]))[0], specific_index[j]]
            error_in = error[np.where((error <= 10) & (error >= -10))[0]]
            # ratio = (error.shape[0] - error_in.shape[0]) / error.shape[0]
            # ratio_all.append(ratio)
            path_output = os.path.join(path_o, str(name_tunnel[i][0]) + '-' + name_error[specific_index[j]])
            mu, sigma = customise_histogram(error_in, path_output)
            # rmse = np.sqrt(np.mean(np.square(error)))
            mae = np.mean(np.abs(error))
            statistics[-1].append(mu)
            statistics[-1].append(sigma)
            statistics[-1].append(mae)
    
    statistics = np.asarray(statistics)
    statistics = pd.DataFrame(statistics)
    statistics.to_excel(os.path.join(path_o, 'statistics.xlsx'), header=False, index=False)
    
    # ratio_all = np.asarray(ratio_all)
    # print(np.max(ratio_all))
    '''all'''
    
    '''joint'''
    joint_errors = errors[np.where(errors[:, -2] != 0)[0], :]
    statistics = []
    
    for i in range(len(name_tunnel)):
        statistics.append([])
        for j in range(len(specific_index)):
            joint_error = joint_errors[np.where(np.isin(joint_errors[:, -1], name_tunnel[i]))[0], specific_index[j]]
            joint_error_in = joint_error[np.where((joint_error <= 10) & (joint_error >= -10))[0]]
            path_output = os.path.join(path_o, 'j' + str(name_tunnel[i][0]) + '-' + name_error[specific_index[j]])
            mu, sigma = customise_histogram(joint_error_in, path_output)
            # rmse = np.sqrt(np.mean(np.square(joint_error)))
            mae = np.mean(np.abs(joint_error))
            statistics[-1].append(mu)
            statistics[-1].append(sigma)
            statistics[-1].append(mae)

    statistics = np.asarray(statistics)
    statistics = pd.DataFrame(statistics)
    statistics.to_excel(os.path.join(path_o, 'statistics_joint.xlsx'), header=False, index=False)
    '''joint'''