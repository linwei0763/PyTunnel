import numpy as np
import os
import pandas as pd

if __name__ == '__main__':
    
    '''config'''
    path_i = '../Seg2Tunnel/seg2tunnel'
    
    tunnel_no = ['1', '2']
    '''config'''
    
    '''call'''
    files = os.listdir(path_i)
    
    rings = {}
    
    for tunnel in tunnel_no:
        rings[tunnel] = []
    
    for file in files:
        tunnel = file.split('-')[0]
        ring = file.split('.')[0].split('-')[-1]
        if tunnel in rings.keys():
            rings[tunnel].append(ring)
    '''call'''
            
    
    