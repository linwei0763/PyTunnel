import numpy as np
import os


if __name__ == '__main__':
    
    '''------config------'''
    path_i = 'data'
    num_epoch = 100
    '''------config------'''
    
    '''------run------'''
    training_loss = np.zeros(num_epoch)
    test_loss = np.zeros(num_epoch)
    oa = np.zeros(num_epoch)
    miou = np.zeros(num_epoch)
    
    with open(os.path.join(path_i, 'log_train.txt'), 'r') as f:
        logs = f.readlines()

    for log in logs:
        if '**** EPOCH' in log:
            count_epoch = int(log.split(' ')[-2])
        elif 'training set loss' in log:
            training_loss[count_epoch] = float(log.split(':')[-1])
        elif 'validation set loss' in log:
            test_loss[count_epoch] = float(log.split(':')[-1])
        elif 'OA' in log:
            oa[count_epoch] = float(log.split(' ')[-1])
        elif 'mIoU' in log:
            miou[count_epoch] = float(log.split(' ')[-1])
    '''------run------'''