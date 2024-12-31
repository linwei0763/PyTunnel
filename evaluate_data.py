import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix


def calculate_metric(labels, preditions):
    
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
    miou = sum(iou_list) / float(num_class)

    correct = np.sum(labels == predictions)
    total_seen = len(labels)
    oa = correct / total_seen
    
    return oa, miou, iou_list, conf_matrix


if __name__ == '__main__':
    
    '''------config------'''
    path_i = 'data'
    
    index_label = 4
    index_prediction = 5
    index_confidence = 6
    num_class = 8
    
    flag_joint = False
    if flag_joint:
        index_joint = -1
    '''------config------'''
    
    '''------run------'''
    files = os.listdir(path_i)
    distance = {}
    oa = {}
    miou = {}
    mean_confidence = {}
    d_o_m = []
    
    pc_all = []
    
    for file in files:
        pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
        pc = np.asarray(pc)
        if flag_joint:
            pc = pc[np.where(pc[:, index_joint] != 0)[0], :]
        
        xy = pc[:, 0:2]
        labels = pc[:, index_label]
        predictions = pc[:, index_prediction]
        confidence = pc[:, index_confidence]
        
        distance[file] = np.linalg.norm(np.mean(xy, axis=0))
        oa[file], miou[file], _, _ = calculate_metric(labels, predictions)
        mean_confidence[file] = np.mean(confidence)
        d_o_m.append([distance[file], oa[file], miou[file]])
        
        pc_all.append(pc)
    
    d_o_m = np.asarray(d_o_m)
    
    pc_all = np.vstack(pc_all[:])
    labels = pc_all[:, index_label]
    predictions = pc_all[:, index_prediction]
    confidence = pc_all[:, index_confidence]
    
    oa_all, miou_all, iou_list_all, conf_matrix_all = calculate_metric(labels, predictions)
    mean_confidence_all = np.mean(confidence)
    
    print(oa_all)
    print(miou_all)
    print(iou_list_all)
    print(mean_confidence_all)
    print(np.percentile(confidence, 10))
    '''------run------'''