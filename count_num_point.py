import numpy as np
import os
import pandas as pd


if __name__ == '__main__':
    
    '''------config------'''
    path_i = '../Seg2Tunnel/seg2tunnel'    
    tunnel_no = ['4', '5']
    '''------config------'''
    
    count = 0
    files = os.listdir(path_i)
    for file in files:
        if file.split('-')[0] in tunnel_no:
            pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
            pc = np.asarray(pc)
            count += pc.shape[0]
    print(count)