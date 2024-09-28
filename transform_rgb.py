#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:06:41 2024

@author: linwei
"""

import numpy as np
import os
import pandas as pd


path_input = 'data'
path_output = 'result'
if not os.path.exists(path_output):
    os.makedirs(path_output)

index_scale = [4,]
label = [
    [1, 2, 3, 4, 5, 6, 7],
    ]
label_r = [
    [253, 144, 53, 33, 49, 68, 68],
    ]
label_g = [
    [231, 215, 183, 145, 104, 67, 1],
    ]
label_b = [
    [37, 67, 121, 140, 142, 131, 84],
    ]

files = os.listdir(path_input)
pc_all = []
for file in files:
    pc = pd.read_csv(os.path.join(path_input, file), sep=' ', header=None)
    pc = np.asarray(pc)
    pc_all.append(pc)
pc_all = np.vstack(pc[:])
pc = pc_all

xyz = pc[:, 0:3]
scale = pc[:, index_scale]
scale_r = np.interp(scale, label, label_r).reshape(-1, 1)
scale_g = np.interp(scale, label, label_g).reshape(-1, 1)
scale_b = np.interp(scale, label, label_b).reshape(-1, 1)
rgb = np.hstack((scale_r, scale_g, scale_b))
