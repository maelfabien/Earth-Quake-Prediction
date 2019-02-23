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
    
def sub_table(index, seg_len):
    """This function select a subtable from the main earthquake table"""
    
    #We keep only the acoustic_data as description variable to save memory
    X = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16},
        usecols=['acoustic_data'], skiprows =range(1, index-1), nrows=seg_len)
        
    #We pick up the last 'time_to_failure' as a prediction variable
    y = pd.read_csv('../input/train.csv', dtype={'time_to_failure': np.float32},
        usecols=['time_to_failure'], skiprows =range(1, index-1+seg_len), nrows=1).values[0][0]
        
    return (X,y)

def mk_features(X):
    
    """Creation of features for ML algorithms"""
    
    feats = {}
    feats['mean'] = [X.mean()]
    feats['std'] = [X.std()]
    feats['max'] = [X.max()]
    feats['min'] = [X.min()]
    
    feats['mean_change_abs'] = [np.diff(X, axis=0).mean()]
    feats['mean_change_rate'] = [np.nonzero(np.diff(X, axis=0) / (X[:-1]+0.00000001))[0].mean()]
    feats['abs_max'] = [np.abs(X).max()]
    feats['abs_min'] = [np.abs(X).min()]
    
    for j in [50000,10000,5000]:
        for i in range(int(len(X)/j)):
            X_ = X[i*j:(i+1)*j]
            feats[f'std_{j}_{i}'] = [X_.std()]
            feats[f'mean_{j}_{i}'] = [X_.mean()]
            feats[f'max_{j}_{i}'] = [X_.max()]
            feats[f'min_{j}_{i}'] = [X_.min()]
            feats[f'mean_change_rate_{j}_{i}'] = [np.nonzero(np.diff(X_, axis=0) / (X_[:-1]+0.00000001))[0].mean()]
    
    feats['max_to_min'] = [X.max()/np.abs(X.min())]
    feats['max_to_min_diff'] = [X.max() - np.abs(X.min())]
    
    for i in  [10,20,50,100,300,500]:
        feats[f'count_{i}'] = [len((X[np.where(np.abs(X)>i)]))]
        
    feats['sum'] = [X.sum()]
    
    for q in [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,0.999]:
        feats[f'quantile_{q}'] = [np.quantile(X,q)]
        
    return pd.DataFrame(feats)


import time
ind= gen_index(150000)
X=[]
Y=[]
j=0
for i in ind :
    t1 = time.time()
    X.append(mk_features(df[i:i+150000,0]))
    Y.append(df[i+1,1])
    t2 = time.time()
    print(f'{j}_sub_table : {round(t2-t1,1)}')
    j+=1

train = pd.concat(X)
train['y'] = Y

"""Xtrain, Xtest, ytrain, ytest = train_test_split(train.drop('y',axis=1), train['y'], train_size=0.9)

for n in [0.01,0.05,0.1,0.5,1]:
    print('**********'+str(n)+'*************')
    for lamb in [0.001,0.003,0.005,0.1,0.20]:
        model = XGBRegressor(eta=n, max_depth = 5, reg_lambda = lamb)
        model.fit(Xtrain,ytrain)
        print(f"{lamb} : {round(mean_absolute_error(model.predict(Xtest),ytest),2)}")"""


#Final model
model = XGBRegressor(eta=0.05, max_depth = 5, reg_lambda = 0.005).fit(train.drop('y',axis=1),train['y'])

test = []
#We load all test sets in a list
for file in glob.glob("../input/test/*.csv"):
    test.append(file.split('/')[-1].split('.')[0])

Test = []
for csv in test:
    Test.append(mk_features(pd.read_csv(f'../input/test/{csv}.csv').values))
    print(str(round(len(Test)/len(test)*100)) + ' %')
    
Xtest = pd.concat(Test)

pred = model.predict(Xtest)
submit = pd.DataFrame(test)
submit = submit.rename(columns={0:'seg_id'})
submit['time_to_failure'] = pred
submit.to_csv('submission.csv', index=False)
