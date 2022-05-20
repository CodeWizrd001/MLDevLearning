import pandas as pd
import numpy as np
import random
import torch

from sklearn.preprocessing import LabelEncoder

DATA_DIR = 'data/book/'
DATA_PATH = DATA_DIR + 'dummy.csv'
data = pd.read_csv(DATA_PATH,index_col='id')

authors = data['Author'].unique()
# authors = np.array(authors)
authorEncoder = LabelEncoder()
encAuth = authorEncoder.fit_transform(authors)

genres = data['Genre'].unique()
# genres = np.array(genres)
genreEncoder = LabelEncoder()
encGenre = genreEncoder.fit_transform(genres)

data['Author'] = authorEncoder.transform(data['Author'])
data['Genre'] = genreEncoder.transform(data['Genre'])

data = data.drop(['ISBN','Name'],axis=1)

def gen_data(data_main,samples=100,shuffle=True):
    data = data_main.copy()
    data = []
    labels = []
    for i in range(samples):
        if shuffle:
            data = data.sample(frac=1)
        # Choose random row from dataframe
        row = data.iloc[random.randint(0,len(data)-1)]
        label = row['Genre']
        