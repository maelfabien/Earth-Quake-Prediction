# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, Flatten, Dropout, Activation
from keras import Model
from keras import backend as K
import keras.metrics

import glob, os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

events = np.array([  5656574,  50085878, 104677356, 138772453, 187641820, 218652630,
      245829585, 307838917, 338276287, 375377848, 419368880, 461811623,
      495800225, 528777115, 585568144, 621985673])

df = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16,'time_to_failure': np.float32}).values

def gen_index(seg_len):
    """This function generate a list of initial value for the splitting of the dataset"""
    
    #Initiation of the list of index
    list_index = []
    
    #Number of tables that we can fit between two indexes
    num_tables = int(np.floor(events[0])/seg_len)
    
    #Total number of lines we have has a marges
    tot_lines = events[0]-seg_len*num_tables
    
    
    #Minimum index, this is the index of previous earthquake
    ind_min = 0
    
    #This loop generate all the indexes between two indexes
    for i in range(num_tables):
        
        #If we have spare lines, we randomize a bit the index we choose
        if tot_lines:
            u = random.randint(0,int(tot_lines/10))
            tot_lines -= u
        else:
            u = 0
        
        #We add the randomized index to the current index
        ind_min +=u
        
        #We add the index to the list
        list_index.append(ind_min)
        
        #We update the index based on the length of the data
        ind_min += seg_len
        
    #We make the same, but this time we can loop over a window between two indexes
    for i in range(1,len(events)):
        #Count number of table to make
        num_tables = int(np.floor((events[i]-events[i-1])/seg_len))
        tot_lines = (events[i]-events[i-1]) - seg_len*num_tables
        ind_min = events[i-1]
        for i in range(num_tables):
            if tot_lines:
                u=random.randint(0,int(tot_lines/10))
                tot_lines-= u
            else:
                u = 0
            ind_min += u
            list_index.append(ind_min)
            ind_min += seg_len
            
    #We return the list generated        
    return np.array(list_index)
    
ind= gen_index(150000)
Y = df[ind+1,1]

###Creation of a LSTM using Keras
inputs = Input(shape=(150000,1), name = 'input')
x = LSTM(256, return_sequences=True)(inputs)
x = Dropout(0.3)(x)
x = LSTM(256, input_shape=(20, 256))(x)
x = Dropout(0.3)(x)
x = Dense(256)(x)
x = Dropout(0.3)(x)
predict = Dense(1, activation='linear')(x)
model = Model(inputs =inputs, outputs = predict)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=[keras.metrics.mae])
[Y[j]]
for j in range(len(ind)):
    a = np.reshape(df[ind[j]:ind[j]+150000,0], (1,150000,-1))
    model.fit(a, [[Y[j]]], epochs=1, batch_size=1)
    print(j)