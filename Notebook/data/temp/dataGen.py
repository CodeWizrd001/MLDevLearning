import pandas as pd
import numpy as np
import random

increasing = np.array([0,1,2,3,4,5,6,7,8,9])
decreasing = np.array([9,8,7,6,5,4,3,2,1,0])

INC_FOLDER = '1/'
DEC_FOLDER = '0/'

def genData(samples = 100):
    data = []
    for i in range(samples):
        print(f'[+] Generating sample {i}')
        offset = np.random.rand(10) / 5
        inc = increasing + offset
        with open(INC_FOLDER + str(i) + '.csv', 'w') as f:
            for j in inc:
                f.write(str(j) + '\n')
        offset = np.random.rand(10) / 5
        dec = decreasing + offset
        with open(DEC_FOLDER + str(i) + '.csv', 'w') as f:
            for j in dec:
                f.write(str(j) + '\n')

genData(100)

def read_data() :
    IDATA_DIR = '1/'
    DDATA_DIR = '0/'

    idata = []
    ddata = []
    
    for f in os.listdir(IDATA_DIR):
        df = pd.read_csv(IDATA_DIR + f, header=None)
        idata.append(df.values)
    iLabels = [1 for i in range(len(idata))]

    for f in os.listdir(DDATA_DIR):
        df = pd.read_csv(DDATA_DIR + f, header=None)
        ddata.append(df.values)
    dLabels = [0 for i in range(len(ddata))]

    data = idata + ddata
    labels = iLabels + dLabels

    data = np.array(data)
    labels = np.array(labels)

    return data, labels