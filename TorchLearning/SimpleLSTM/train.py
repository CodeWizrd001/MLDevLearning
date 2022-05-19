import numpy as np
import pandas as pd
import torch
import random 

import lstm
import data

d , l = data.gen_random_data(100)

def train(model:lstm.LSTM,optimizer,data,labels,epochs=10) :
    model.train()
    for epoch in range(1,epochs+1):
        x = torch.from_numpy(data).float()
        y = torch.from_numpy(labels).float()
        output = model(x)
        loss = torch.nn.functional.binary_cross_entropy(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'[+] Epoch : {epoch:<5d} :::   Loss : {loss.item():<10.5f}')

if __name__ == '__main__':
    lstm = lstm.LSTM(21,10,1)
    optimizer = torch.optim.Adam(lstm.parameters(),lr=0.01)

    train(lstm,optimizer,d,l,epochs=500)

    # Save the model
    torch.save(lstm.state_dict(),'model/model.pt')
    print('[+] Model Saved')