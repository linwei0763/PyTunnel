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
    
    plt.figure(figsize=(4.5 / 2.54, 4.5 / 2.54), dpi=600)
    
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    '''------'''
    
    '''------'''
    plt.xlim(-4, 4)
    plt.ylim(0, 0.2)
    
    plt.xlabel('Error (mm)')
    plt.ylabel('Frequency')
    
    weights = np.ones_like(data) / float(len(data))
    
    plt.hist(data, bins=25, range=(-4, 4), weights=weights, color='#a9a9a9')
    
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
    '''config'''
    file_gt = 'gt_dislocation.xlsx'
    path_o = 'result'
    file_dislocation = 'dislocation.xlsx'
    file_o = 'dislocation_comparison.xlsx'
    num_sheet = 3
    '''config'''
    
    gt = {}
    data = pd.read_excel(file_gt, header=None, names=None)
    data = np.asarray(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 2):
            if not np.isnan(data[i, j + 2]):
                joint_no = str(int(data[i, 0])) + '-' + str(int(data[i, 1])) + '-' + str(int(j + 1))
                gt[joint_no] = data[i, j + 2]
    
    dislocation = {}
    data = pd.read_excel(os.path.join(path_o, file_dislocation), header=None, names=None)
    data = np.asarray(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 3):
            if not np.isnan(data[i, j + 3]):
                joint_no = str(int(data[i, 0])) + '-' + str(int(data[i, 2])) + '-' + str(int(j + 1))
                dislocation[joint_no] = data[i, j + 3] * 1000
    
    data_all = []
    error = []
    for key in gt.keys():
        if key not in dislocation.keys():
            continue
        if gt[key] == 0:
            continue
        data_all.append([int(key.split('-')[0]), int(key.split('-')[1]), int(key.split('-')[2])])
        data_all[-1].append(gt[key])
        data_all[-1].append(dislocation[key])
        error.append(gt[key] - dislocation[key])
    
    error = np.asarray(error)
    mae = np.mean(np.abs(error))
    c = error[np.where(np.abs(error[:] <= 1))[0]].shape[0]
    print('The MAE of %d dislocation data is %.1f mm.' % (error.size, mae))
    print('There are %d dislocation data, the absolute errors of which are smaller than 1 mm, accounting for %.1f%%.' % (c, c / error.shape[0] * 100))
    
    data_all = np.asarray(data_all)
    data_all = pd.DataFrame(data_all)
    data_all.to_excel(os.path.join(path_o, file_o), header=False, index=False)
    
    customise_histogram(error[:], os.path.join(path_o, 'dislocation_error'))