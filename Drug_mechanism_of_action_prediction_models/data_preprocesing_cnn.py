#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:58:41 2020

@author: liuziwei
"""
import h5py
import tables
import time
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from pathlib import Path
import random

#%% Read data

train_data=pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/timeseries_train_dataset_12classes_8signals.csv')
val_data = pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/timeseries_val_dataset_12classes_8signals.csv')

train_label=pd.read_csv(('/Volumes/Ashur Pro2/SyngentaScreen/train_labels_7classes.csv'))
val_label= pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/val_labels_7classes.csv')

#%%

def split_sequences( sequences, n_steps):
  X = list()
  L= len(sequences)
  for i in range(0, L, n_steps):
    seq_x = sequences.iloc[i:i+n_steps, :]
    X.append(seq_x)
  return X


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        mean= df[feature_name].mean()
        result[feature_name] = df[feature_name] - mean
        max_value = result[feature_name].max()
        min_value= result[feature_name].min()
        result[feature_name] = result[feature_name] / (max_value - min_value)
    return result

#%%

train_list= split_sequences(train_data, 876)
val_list = split_sequences(val_data, 876)

dataset= pd.concat([train_data,val_data],axis=0,ignore_index=True )
dataset_normalized= normalize(dataset)
dataset_split=split_sequences(dataset 876)


train_list_normalized=dataset_split[0:11041]
val_list_normalized=dataset_split[11041:12996]


train_label_list =[]
for index, rows in train_label.iterrows():
    train_label_list.append(rows)

val_label_list =[]
for index, rows in val_label.iterrows():
    val_label_list.append(rows)


#shuffle
a = list(zip(train_list_normalized, train_label_list))

random.shuffle(a)

train_list_normalized, train_label_list= zip(*a)

b = list(zip(val_list_normalized, val_label_list))

random.shuffle(b)

val_list_normalized, val_label_list= zip(*b)

#%%
#Fill nans
train_list_fillednans= [train_list_normalized[i].fillna(-1) for i in range(len(train_list_normalized))]
val_list_fillednans= [val_list[i]_normalized.fillna(-1) for i in range(len(val_list_normalized))]

#%% ADD in signals

train_list_fillednans= [train_list_fillednans[i].reset_index(drop=True)for i in range(len(train_list_fillednans))]
val_list_fillednans= [val_list_fillednans[i].reset_index(drop=True) for i in range(len(val_list_fillednans))]

new_signal= [0]* 375 + [0.05]*251 +[0]*250
new_signal= pd.Series(new_signal)

for i in range(len(train_list_fillednans)):
    train_list_fillednans[i].insert(loc=0, column='new_signal',value= new_signal) 

for i in range(len(val_list_fillednans)):
    val_list_fillednans[i].insert(loc=0, column='new_signal',value= new_signal) 

#%%

list_of_foo = [df[np.newaxis,:,:] for df in train_list_fillednans]
bigarray_foo = np.concatenate(list_of_foo, axis=0)


list_of_arrays_val = [df[np.newaxis,:,:] for df in val_list_fillednans]
bigarray_3d_val = np.concatenate(list_of_arrays_val, axis=0)



train_label_merged=pd.DataFrame(train_label_list)
train_label_merged.reset_index(drop=True, inplace=True)
train_label_merged['ts_id']=np.arange(train_label_merged.shape[0]).astype(int) #add in sample id


val_label_merged=pd.DataFrame(val_label_list)
val_label_merged.reset_index(drop=True, inplace=True)
val_label_merged['ts_id']=np.arange(val_label_merged.shape[0]).astype(int)

train_label_merged.drop(['CSN'], axis=1,inplace=True)
val_label_merged.drop(['CSN'], axis=1,inplace=True)

#%% 
#Save to a hdf file

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)


with tables.File('1d_cnn_8signals.hdf', 'w') as fid:
    group = fid.create_group('/','train')
    fid.create_earray(group,
                      'ts_data',obj=bigarray_foo,filters=FILTERS)
    fid.create_table(group,
                     'labels', obj=train_label_merged.to_records(index=False),filters=FILTERS)
    group2= fid.create_group('/','val')
    fid.create_earray(group2,
                      'ts_data',obj=bigarray_3d_val,filters=FILTERS)
    fid.create_table(group2,
                     'labels', obj=val_label_merged.to_records(index=False),filters=FILTERS)

















