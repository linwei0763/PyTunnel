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
    path_i = 'data'
    file_i = 'data.xlsx'
    path_o = 'result'
    num_sheet = 3
    '''config'''
    
    dis_all = []
    real = None
    
    for i in range(num_sheet):
        dis = pd.read_excel(os.path.join(path_i, file_i), sheet_name=i)
        dis = np.asarray(dis)
        if real is None:
            real = dis[:, 2:9]
            real = real.reshape(-1, 1)
        dis = dis[:, 11:18]
        dis = dis.reshape(-1, 1)
        dis_all.append(dis*1000)
    
    data = real
    for i in range(num_sheet):
        data = np.hstack((data, dis_all[i]))
    data = data[~np.isnan(data).any(axis=1)]
    data = data[data[:, 0] != 0]
    
    error = data[:, 0].reshape(-1, 1) - data[:, 1:]
    for i in range(num_sheet):
        c = error[np.where(abs(error[:, i] <= 1))[0], i].shape[0]
        print(c)
        print(c / error.shape[0] * 100)
        customise_histogram(error[:, i], os.path.join(path_o, 'dis_error_' + str(i)))



