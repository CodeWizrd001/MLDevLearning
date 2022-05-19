import numpy as np
import pandas as pd
import torch
import random 

import lstm
import data

DATA_SIZE = 10

d , l = data.gen_random_data(DATA_SIZE)

# Load Model
model = lstm.LSTM(21,10,1)
model.load_state_dict(torch.load('model/model.pt'))
model.eval()

for i in range(2*DATA_SIZE):
    x = torch.from_numpy(d[i]).float().unsqueeze(0)
    y_true = torch.from_numpy(l[i]).float()
    y = model(x)
    print(f'[+] Prediction : {y.item():<10.5f}  :: True : {y_true.item():<10.5f}')