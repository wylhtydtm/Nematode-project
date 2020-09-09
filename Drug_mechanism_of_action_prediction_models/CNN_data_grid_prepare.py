#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:41:18 2020

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


col_names= list(train_data.columns)

def normalize_val(df1, df2): 
    result1= df1.copy()
    result2= df2.copy()
    for feature_name in col_names:
        mean = df1[feature_name].mean()
        result1[feature_name]=df1[feature_name] - mean
        result2[feature_name] = df2[feature_name] - mean
        max_value=result1[feature_name].max()
        min_value=result1[feature_name].max()
        result2[feature_name]= result2[feature_name] / (max_value - min_value)
    return result2
    
        
def mean(df):
    result = df.copy()
    for feature_name in df.columns:
        mean= df[feature_name].mean()
        result[feature_name] = df[feature_name] - mean
    return result
        
#%%


train_data_normalized= normalize(train_data)


train_data_normal= normalise(train_data)


val_data_normalized =val_data.copy()
val_data_normalized['eigen_projection_1']= (val_data['eigen_projection_1']- 0.031350)/(22.178193 +12.655297)
val_data_normalized['eigen_projection_2'] = (val_data['eigen_projection_2']+0.075473)/(22.285016 +12.548473999999999)

val_data_normalized['eigen_projection_3']= (val_data['eigen_projection_3']+0.034836)/(7.17893 + 8.636678)
val_data_normalized['eigen_projection_4'] = (val_data['eigen_projection_4']-0.217049)/(5.3602887 +9.4339155)



val_data_normalized['eigen_projection_5']= (val_data['eigen_projection_5']- 0.003225)/(4.473516300000001 +4.1620556)
val_data_normalized['eigen_projection_6'] = (val_data['eigen_projection_6']-0.231623)/(2.7773072000000005 +4.4563215)
val_data_normalized['eigen_projection_7']= (val_data['eigen_projection_7']- 0.049025)/(2.6315647 +2.7616194999999997)


val_data_normalized['speed'] = (val_data['speed']-18.297362)/(1464.912338 + 1581.977862)
val_data_normalized['angular_velocity']= (val_data['angular_velocity'] + 0.000116)/(10.059385+ 10.108462)


root_dir_Dec= Path('/Volumes/Ashur Pro2/SyngentaScreen')

train_path= os.path.join(root_dir_Dec, 'normalized_timeseries_7classes_wifnanss_eigenprojections_speed_angular_velocity.csv')
train_data_normalized.to_csv(train_path, index=False)

val_path= os.path.join(root_dir_Dec, 'normalized(totrainingset)_validation_rawvtimeseriesdata_7classes_wifnan_eigenprojections_speed_angular_velocity.csv')
val_data_normalized.to_csv(val_path, index=False)

#%%
train_list= split_sequences(train_data_normalized, 876)
val_list = split_sequences(val_data_normalized, 876)

train_label.drop(['CSN'], axis=1,inplace=True)
val_label.drop(['CSN'], axis=1,inplace=True)

train_label_normalized= train_label.copy()
train_label_normalized['imaging_plate_drug_concentration']=(train_label['imaging_plate_drug_concentration']-35.612092)/(164.28790800000002+35.612092)

val_label_normalized= val_label.copy()
val_label_normalized['imaging_plate_drug_concentration']=(val_label['imaging_plate_drug_concentration']-35.612092)/(164.28790800000002+35.612092)


train_label_list =[]
for index, rows in train_label_normalized.iterrows():
    train_label_list.append(rows)

val_label_list =[]
for index, rows in val_label_normalized.iterrows():
    val_label_list.append(rows)


c = list(zip(train_list, train_label_list))

random.shuffle(c)

train_list, train_label_list= zip(*c)

d = list(zip(val_list, val_label_list))

random.shuffle(d)

val_list, val_label_list= zip(*d)

#%%
#Fill nans
train_list_fillednans= [train_list[i].fillna(-1) for i in range(len(train_list))]
val_list_fillednans= [val_list[i].fillna(-1) for i in range(len(val_list))]


train_label_list_1= list(train_label_list)
del train_label_list_1[7079]
list_of_foo = [df[np.newaxis,:,:] for df in train_list]
del list_of_foo[7079]
bigarray_foo = np.concatenate(list_of_foo, axis=0)


list_of_arrays_val = [df[np.newaxis,:,:] for df in val_list]
bigarray_3d_val = np.concatenate(list_of_arrays_val, axis=0)



train_label_merged=pd.DataFrame(train_label_list_1)
train_label_merged.reset_index(drop=True, inplace=True)
train_label_merged['ts_id']=np.arange(train_label_merged.shape[0]).astype(int)

val_label_merged=pd.DataFrame(val_label_list)
val_label_merged.reset_index(drop=True, inplace=True)
val_label_merged['ts_id']=np.arange(val_label_merged.shape[0]).astype(int)

train_label_merged.drop(['CSN'], axis=1,inplace=True)
val_label_merged.drop(['CSN'], axis=1,inplace=True)

#%%

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)


with tables.File('speed_for_markovetransition_new1.hdf', 'w') as fid:
    group = fid.create_group('/','train')
    fid.create_earray(group,
                      'tw_data',obj=bigarray_foo,filters=FILTERS)
    fid.create_table(group,
                     'labels', obj=train_label_merged.to_records(index=False),filters=FILTERS)
    group2= fid.create_group('/','val')
    fid.create_earray(group2,
                      'tw_data',obj=bigarray_3d_val,filters=FILTERS)
    fid.create_table(group2,
                     'labels', obj=val_label_merged.to_records(index=False),filters=FILTERS)




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

def split_series( sequences, n_steps):
  X = list()
  L= len(sequences)
  for i in range(0, L, n_steps):
    seq_x = sequences[i:i+n_steps]
    X.append(seq_x)
  return X

train_data = train_data['speed']
val_data= val_data['speed']

train_data= pd.DataFrame(train_data)
val_data= pd.DataFrame(val_data)

train_list= split_series(train_data, 876)
val_list = split_series(val_data, 876)



train_list= [train_list[i].reset_index(drop=True)for i in range(len(train_list))]
val_list= [val_list[i].reset_index(drop=True) for i in range(len(val_list))]

train_list= [train_list[i].to_numpy() for i in range(len(train_list))]
val_list= [val_list[i].to_numpy() for i in range(len(val_list))]


train_list= [train_list[i].fillna(0) for i in range(len(train_list))]
val_list= [val_list[i].fillna(0) for i in range(len(val_list))]


train_list= [train_list[i].reshape(1,-1)for i in range(len(train_list))]
val_list= [val_list[i].reshape(1,-1) for i in range(len(val_list))]


from pyts.approximation import PiecewiseAggregateApproximation

transformer = PiecewiseAggregateApproximation(window_size=4)


train_list= [transformer.transform(train_list[i]) for i in range(len(train_list))]
val_list = [transformer.transform(val_list[i]) for i in range(len(val_list))]



train_list= [train_list[i].transpose()for i in range(len(train_list))]
val_list= [val_list[i].transpose() for i in range(len(val_list))]



#%%

def feature_scaling_datasets(ts_datasets):
    normalized_ts_datasets = []

    for ts in ts_datasets:
        normalized_ts_datasets.append(feature_scaling(ts))

    return normalized_ts_datasets

def feature_scaling(ts):
    n = len(ts)
    maximum = max(ts)
    minimum = min(ts)

    normalized_ts = ts.copy()
    r = maximum-minimum
    for i in range(n):
        normalized_ts[i] = (ts[i]-minimum)/r

    return normalized_ts


x_trains = feature_scaling_datasets(train_list)
x_val = feature_scaling_datasets(val_list)

x_train_fillednans= [x_trains[i].fillna(0) for i in range(len(x_trains))]
x_val_fillednans= [x_val[i].fillna(0) for i in range(len(x_val))]

x_train_fillednans= [x_train_fillednans[i].tolist() for i in range(len(x_train_fillednans))]
x_val_fillednans= [x_val_fillednans[i].tolist() for i in range(len(x_val_fillednans))]
                      
#%%

class Grid:
    def __init__(self, m=5, n=5):
        self.m = m
        self.n = n

    #find the best parameters(m, n) for matrix representation using train datasets
    def train(self, x_trains, y_trains):
        best_m, best_n = self.step1(x_trains, y_trains)
        best_m, best_n = self.step2(x_trains, y_trains, best_m, best_n)
        self.m = best_m
        self.n = best_n

    def step1(self, x_trains, y_trains):
        min_error_rate = sys.float_info.max

        best_m = 5
        best_n = 5

        for m in range(5, 40, 5):
            for n in range(5, 35, 5):
                self.m = m
                self.n = n
                train_matrices = self.dataset2Matrices(x_trains)
                error_rate = loocv(train_matrices, y_trains)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_m = m
                best_n = n

        return best_m, best_n

    def step2(self, x_trains, y_trains, center_m, center_n):
        min_error_rate = sys.float_info.max

        for m in range(center_m-4, center_m+5):
            for n in range(center_n-4, center_n+5):
                self.m = m
                self.n = n
                train_matrices = self.dataset2Matrices(x_trains)
                error_rate = loocv(train_matrices, y_trains)

            if error_rate < min_error_rate:
                min_error_rate = error_rate
                best_m = m
                best_n = n
                self.x_matrices_train = train_matrices

        return best_m, best_n


    def dataset2Matrices(self, ts_set):
        matrices = []
        for ts in ts_set:
            matrices.append(self.ts2Matrix(ts))

        return np.array(matrices)

    def ts2Matrix(self, ts):
        matrix = np.zeros((self.m, self.n))
        T = len(ts)

        height = 1.0/self.m
        width = T/self.n

        for idx in range(T):
            i = int((1-ts[idx])/height)
            if i == self.m:
                i -= 1

            t = idx+1
            j = t/width
            if int(j) == round(j, 7):
                j = int(j)-1
            else:
                j = int(j)

            matrix[i][j] += 1
        return matrix

#%%

g = Grid(80, 200)

x_matrices_train = g.dataset2Matrices(x_train_fillednans)
x_matrices_test = g.dataset2Matrices(x_val_fillednans)


import matplotlib.pylab as plt

# visualize time series representation sample
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(x)
ax2.imshow(x_matrices_train[1], interpolation='nearest', cmap='gray', aspect='auto')
plt.show()


plt.figure()
ax.plot(x)





