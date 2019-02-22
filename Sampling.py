# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

events = np.array([  5656574,  50085878, 104677356, 138772453, 187641820, 218652630,
      245829585, 307838917, 338276287, 375377848, 419368880, 461811623,
      495800225, 528777115, 585568144, 621985673])

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
    
ind= gen_index(150000)
(X,y) = sub_table(ind[0],150000)