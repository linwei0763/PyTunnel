import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    
    '''------config------'''
    path_i = 'data'
    
    index_label = 4
    index_prediction = 5
    num_class = 2

    '''------config------'''
    
    '''------run------'''
    files = os.listdir(path_i)
    oa = {}
    miou = {}
    
    for file in files:
        pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
        pc = np.asarray(pc)
        
        labels = pc[:, index_label]
        predictions = pc[:, index_prediction]
        
        conf_matrix = confusion_matrix(labels, predictions, labels=np.arange(0, num_class, 1))
        gt_classes = np.sum(conf_matrix, axis=1)
        positive_classes = np.sum(conf_matrix, axis=0)
        true_positive_classes = np.diagonal(conf_matrix)
        iou_list = []
        for n in range(0, num_class, 1):
            if float(gt_classes[n] + positive_classes[n] - true_positive_classes[n]) != 0:
                iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
                iou_list.append(iou)
            else:
                iou_list.append(0.0)
        miou[file] = sum(iou_list) / float(num_class)

        correct = np.sum(labels == predictions)
        total_seen = len(labels)
        oa[file] = correct / total_seen
    '''------run------'''