t = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
f = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]

import numpy as np
import pandas as pd
import torch
import random 

RAND_SCALE = 0.05

def get_random_offset():
    return random.choice([1,-1])*random.random()*RAND_SCALE

def gen_random_data(size=10) :
    t_data = []
    f_data = []
    for _ in range(size):
        t_ = [get_random_offset()+i for i in t]
        f_ = [get_random_offset()+i for i in f]
        t_data.append(t_)
        f_data.append(f_)
    t_data = np.array(t_data)
    f_data = np.array(f_data)
    data = np.concatenate((t_data,f_data),axis=0)
    t_lables = np.array([1 for _ in range(size)])
    f_lables = np.array([0 for _ in range(size)])
    labels = np.concatenate((t_lables,f_lables),axis=0)
    labels = labels.reshape(2*size,1)
    return data,labels