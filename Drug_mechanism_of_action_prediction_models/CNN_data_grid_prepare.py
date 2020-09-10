#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:41:18 2020
 PiecewiseAggregateApproximation and Grid transfromation for speed univariate timeseries for classification
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
import matplotlib.pylab as plt
from pyts.approximation import PiecewiseAggregateApproximation
from Grid import Grid
#%%

train_data=pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/7_classes_eigenprojections_speed-angualrvelocity/timeseries_7classes_wifnanss_eigenprojections_speed_angular_velocity.csv')
val_data = pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/7_classes_eigenprojections_speed-angualrvelocity/validation_rawvtimeseriesdata_7classes_wifnan_eigenprojections_speed_angular_velocity.csv')

train_label=pd.read_csv(('/Volumes/Ashur Pro2/SyngentaScreen/7_classes_eigenprojections_speed-angualrvelocity/train_labels_7classes.csv'))
val_label= pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/7_classes_eigenprojections_speed-angualrvelocity/val_labels_7classes.csv')
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
val_list_fillednans= [val_list_normalized[i].fillna(-1) for i in range(len(val_list_normalized))]

train_label_merged=pd.DataFrame(train_label_list)
train_label_merged.reset_index(drop=True, inplace=True)
train_label_merged['ts_id']=np.arange(train_label_merged.shape[0]).astype(int)

val_label_merged=pd.DataFrame(val_label_list)
val_label_merged.reset_index(drop=True, inplace=True)
val_label_merged['ts_id']=np.arange(val_label_merged.shape[0]).astype(int)

train_label_merged.drop(['CSN'], axis=1,inplace=True)
val_label_merged.drop(['CSN'], axis=1,inplace=True)


#Extract speed only

def split_series( sequences, n_steps):
  X = list()
  L= len(sequences)
  for i in range(0, L, n_steps):
    seq_x = sequences[i:i+n_steps]
    X.append(seq_x)
  return X

train_data= pd.DataFrame(train_data['speed'])
val_data= pd.DataFrame(val_data['speed'])

train_list= split_series(train_data, 876)
val_list = split_series(val_data, 876)


#%%


train_list= [train_list[i].reset_index(drop=True)for i in range(len(train_list))]
val_list= [val_list[i].reset_index(drop=True) for i in range(len(val_list))]

train_list= [train_list[i].to_numpy() for i in range(len(train_list))]
val_list= [val_list[i].to_numpy() for i in range(len(val_list))]

train_list= [train_list[i].reshape(1,-1)for i in range(len(train_list))]
val_list= [val_list[i].reshape(1,-1) for i in range(len(val_list))]


transformer = PiecewiseAggregateApproximation(window_size=4)


train_list= [transformer.transform(train_list[i]) for i in range(len(train_list))]
val_list = [transformer.transform(val_list[i]) for i in range(len(val_list))]


train_list= [train_list[i].transpose()for i in range(len(train_list))]
val_list= [val_list[i].transpose() for i in range(len(val_list))]




#%%

def feature_scaling(ts):
    n = len(ts)
    maximum = max(ts)
    minimum = min(ts)

    normalized_ts = ts.copy()
    r = maximum-minimum
    for i in range(n):
        normalized_ts[i] = (ts[i]-minimum)/r

    return normalized_ts


def feature_scaling_datasets(ts_datasets):
    normalized_ts_datasets = []

    for ts in ts_datasets:
        normalized_ts_datasets.append(feature_scaling(ts))

    return normalized_ts_datasets



x_trains = feature_scaling_datasets(train_list)
x_val = feature_scaling_datasets(val_list)


#%%

g = Grid(80, 200)

x_matrices_train = g.dataset2Matrices(x_train_fillednans)
x_matrices_test = g.dataset2Matrices(x_val_fillednans)


# visualize time series representation sample
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(x)
ax2.imshow(x_matrices_train[1], interpolation='nearest', cmap='gray', aspect='auto')
plt.show()





