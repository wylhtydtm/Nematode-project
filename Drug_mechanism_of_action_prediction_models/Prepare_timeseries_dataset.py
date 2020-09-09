#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: liuziwei

 Data preprocssing: 
     
Feats selected bassed on PCA and Pearson's correlation matrix, 
Feats selected(7 in total)='length','speed','speed_head_base','speed_neck','curvature_midbody',
'relative_to_body_angular_velocity_neck','relative_to_body_angular_velocity_hips']
1.For each drug class, create a new df with timestamp, features, unique_id,drug concentration
2.Filter out the worms being tracked for less than 500 timestamps
3.Fill in the nans over 125 timestamps(5s gap) by linear interpolation.
4.Chunk the time window around the first blue light stimulus only,length= 876 timestamps
5.Filter the worms being tracked for the entire process of pre-sti, first bluelight and post-sti,876 timestamps
6.Normalized the dataset to have a mean of 0, maximum1 and minimum -1 for each feature, regardless of their targets
6.Fill in theremaining nans with -1 (only curvature_midbody containing about #0.013% 0 values)
7. Add in stimulation signal in , with 0 (no response) and -1 as assumed response.
8. Shuffile and split the dataset  into (80% train, 10% valdation, 10% test)
7.Write hires_df_class to a hdf5 file  


As the datasets now consist of multivariate timesereis, we define the input dataset(X_train) as a tensor shape
3 dimensions (worm,id, [timestamp(rows),[ features(columns)]]) 
worm_id is the number of samples in the dataset, timestamp, is the constant length of time steps among all the variables, 
features is the 8 variables processed per time step.
 
y_train( labels, drug class in the hire_df)
"""
import h5py
import tables
import time
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from pathlib import Path
from tierpsytools_read_data_get_timeseries import read_timeseries
from helper import (read_metadata,load_bluelight_timeseries_from_results,make_well_id,filter_timeseries)
import random
from sklearn.model_selection import train_test_split


#%% where raw data and meta is stored for Dec

root_dir_Dec = Path('/Volumes/Ashur Pro2/SyngentaScreen')
resNN_dir_Dec = root_dir_Dec / 'Results_NN'

'''
#updated metadata with only N2 strain and mode of action added on 
metadata= pd.read_csv(root_dir_Dec /'metadata_alldrugs_N2.csv')
metadata['imaging_plate_drug_concentration'].isnull().sum()

#replace -1( Nocompound) to 25
metadata['MOA_group'].replace({-1.0: 25.0},inplace=True)

test_meta= pd.read_csv('/Users/liuziwei/Downloads/test_compounds.csv')
test_drug = list(test_meta['drug_type'].unique())

meta_test = metadata[metadata['drug_type'].isin(test_drug)]


meta_test_path= os.path.join(root_dir_Dec, 'meta_test_new.csv')
meta_test.to_csv(meta_test_path, index=False)

train_meta= pd.read_csv('/Users/liuziwei/Downloads/train_compounds.csv')
train_drug = list(train_meta['drug_type'])

meta_train= metadata[metadata['drug_type'].isin(train_drug)]
meta_train_path= os.path.join(root_dir_Dec, 'meta_train_new.csv')
meta_train.to_csv(meta_train_path, index=False)
'''
#%%

root_dir_Jan = Path('/Volumes/Ashur Pro2/SyngentaStrainScreen')
resNN_dir_Jan = root_dir_Jan / 'Results_NN'

metadata_JanFeb= pd.read_csv('/Volumes/Ashur Pro2/SyngentaStrainScreen/metadata_JanFeb_N2.csv')
#replace -1( Nocompound) to 25
metadata_JanFeb['MOA_group'].replace({-1.0: 25.0},inplace=True)
metadata_JanFeb['MOA_group'].isnull().sum()
'''
meta_test_1 = metadata_JanFeb[metadata_JanFeb['drug_type'].isin(test_drug)]

meta_train_1 = metadata_JanFeb[metadata_JanFeb['drug_type'].isin(train_drug)]

meta_test_path_1= os.path.join(root_dir_Jan, 'meta_Jan_test_new.csv')
meta_test_1.to_csv(meta_test_path_1, index=False)
meta_train_path_1= os.path.join(root_dir_Jan, 'meta__Jan_train_new.csv')
meta_train_1.to_csv(meta_train_path_1, index=False)
'''
#%%
meta_train_Dec= pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/meta_train_new.csv')

meta_train_Jan = pd.read_csv('/Volumes/Ashur Pro2/SyngentaStrainScreen/meta__Jan_train_new.csv')

#%%
meta_test_Dec= pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/meta_test_new.csv')

meta_test_Jan = pd.read_csv('/Volumes/Ashur Pro2/SyngentaStrainScreen/meta_Jan_test_new.csv')
#%%
drug_class=list(set(meta_train_Dec['MOA_group'])) #12 classes 

def select_meta(df, filter_dic):
    df_=df.copy()
    for k, v in filter_dic.items():
        df_=df_[df_[k]==v]
        
    return df_


for drug in drug_class:
   globals()['Jan_selected_train_meta_%s' % int(drug)]= select_meta(meta_train_Jan,{'MOA_group':drug})
   globals()['Jan_selected_train_meta_%s' % int(drug)].reset_index(drop= True, inplace=True)
   globals()['Jan_selected_train_meta_%s' % int(drug)].drop(columns=['Unnamed: 5'], inplace=True)
 
#drugclass_1=select_meta(metadata_JanFeb, {'MOA_group': 1.0})
#%%
ID_COLS = ['well_name','imaging_plate_id','date_yyyymmdd']

CATEG_COLS = ID_COLS + ['worm_strain']

HIRES_COLS = ['worm_index', 'timestamp','length','speed','speed_head_base','speed_neck','curvature_midbody',
              'relative_to_body_angular_velocity_neck','relative_to_body_angular_velocity_hips']

feats=['length','speed','speed_head_base','speed_neck','curvature_midbody',
       'relative_to_body_angular_velocity_neck','relative_to_body_angular_velocity_hips']

SAVED_COlS=['timestamp','length','speed','speed_head_base','speed_neck','curvature_midbody',
            'relative_to_body_angular_velocity_neck','relative_to_body_angular_velocity_hips','unique_id','MOA_specific','CSN','imaging_plate_drug_concentration','MOA_group']


def load_bluelight_timeseries_from_drugclass(
        metadata_df,results_dir,saveto=None):

    hires_df = []  # this will have features for one drug class
    from pandas.api.types import CategoricalDtype
    cat_types = {}
    for cat in CATEG_COLS:
        vals = metadata_df[cat].unique()
        cat_types[cat] = CategoricalDtype(categories=vals, ordered=False)
        metadata_df['date_yyyymmdd'] = metadata_df['date_yyyymmdd'].apply(str)
    vals = make_well_id(metadata_df).unique()
    cat_types['well_id'] = CategoricalDtype(categories=vals, ordered=False)

    # so group metadata by video. so far, let's use the bluelight only
    metadata_g = metadata_df.groupby('imgstore_name_blue') 
   
    # loop on video, select which wells need to be read
    for gcounter, (gname, md_group) in tqdm(enumerate(metadata_g)):
        # checks and extract variables out
        md_group[CATEG_COLS]
        filename = results_dir / gname / 'metadata_featuresN.hdf5'
        wells_to_read = list(md_group['well_name']) # Select the file to read
        # read data
        data = read_timeseries(filename, only_wells=wells_to_read)
        # filter bad worm trajectories
        data = filter_timeseries(data)
        data = data[HIRES_COLS+ ['well_name']]
                
        data = pd.merge(data, md_group[CATEG_COLS], how='left', on='well_name')
        #data['date_yyyymmdd'] = data['date_yyyymmdd'].astype(str)
        
        hires_data = data[CATEG_COLS + HIRES_COLS].copy()
    
        # create a single unique well id
        hires_data['well_id'] = make_well_id(hires_data)

        # set some columns to categorical for memory saving
        for col in CATEG_COLS:
            hires_data[col] = hires_data[col].astype(cat_types[col])

        if saveto is None:
            # now grow list
            hires_df.append(hires_data)
            
        else:
            # save to disk
            is_append = False if (gcounter == 0) else True
            
            imaging_plate_id = get_value_from_const_column(md_group,
                                                           'imaging_plate_id')
            hires_data.to_hdf(
                saveto, 'hires_df_{}'.format(imaging_plate_id),
                format='table', append=True,
                min_itemsize={'well_id': 30},
                data_columns=True)

    if saveto is None:
        # join in massive dataframe
        hires_df = pd.concat(hires_df, axis=0, ignore_index=True)
        # fix types
        categ_cols = CATEG_COLS + ['well_id']
        for col in categ_cols:
            hires_df[col] = hires_df[col].astype('category')
        hires_df['unique_id'] = 0  
        df= hires_df.groupby(by=['date_yyyymmdd','imaging_plate_id','well_name', 'worm_index'])['unique_id'].transform(lambda x: len(x)>1)
        hires_df.loc[df,'unique_id'] = hires_df.loc[df,['date_yyyymmdd','imaging_plate_id','well_name', 'worm_index']].astype(str).sum(1).factorize()[0] + 1
        hires_df= pd.merge(hires_df, metadata_df)
        hires_df= hires_df[SAVED_COlS]
        # out
        return  hires_df

#%%
def preprocess_timeseries_three_stimuli(df):
    df = df.groupby('unique_id').filter(lambda g: len(g) >500)
    df_ =[]
    labels=[]
    for k, worm_idx in df.groupby('unique_id'):
        #df_tw=worm_idx.interpolate(method='linear', limit_direction='forward',limit=125, axis=0)# 876 total
        df_tw=worm_idx.interpolate(method='linear',limit=125,axis=0)
        df_tw_1= df_tw.query('1250 <= timestamp <= 2125')   
        if len(df_tw_1)== 876:
            df_.append(df_tw_1.iloc[:,1:10])
            labels.append(df_tw_1.iloc[0,12:15])          
        df_tw_2 = df_tw.query('3750 <= timestamp <= 4625')
        if len(df_tw_2) == 876:
            df_.append(df_tw_2.iloc[:,1:10])
            labels.append(df_tw_2.iloc[0,12:15])
        df_tw_3 = df_tw.query('6250 <= timestamp <= 7125')
        if len(df_tw_3)== 876:
            df_.append(df_tw_3.iloc[:,1:10])
            labels.append(df_tw_3.iloc[0,12:15])
   # agg= pd.concat(df_, axis=0,ignore_index=True)
    # if agg.isnull().any().any():
    #     print('Nans present')
        #agg.fillna(agg.groupby('timestamp').transform('mean'),inplace=True) 
    return df_ ,labels


#%%

hires_df_2= load_bluelight_timeseries_from_drugclass(Dec_selected_meta_2,resNN_dir_Dec,saveto=None)

hires_df_2_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_meta_2,resNN_dir_Jan,saveto=None)

hires_df_2_merged= pd.concat([hires_df_2, hires_df_2_Jan], axis=0, ignore_index=True)

data_2_Dec, labels__2_Dec=preprocess_timeseries_three_stimuli(hires_df_2)

data_2_Jan, labels__2_Jan=preprocess_timeseries_three_stimuli(hires_df_2_Jan)

data_2_final= data_2_Dec + data_2_Jan

labels_2 = labels__2_Dec + labels__2_Jan


c = list(zip(data_2_final, labels_2))

random.shuffle(c)

data_2_final, labels_2 = zip(*c)

X_train, X_test, y_train, y_test = train_test_split(data_2_final, labels_2, test_size=0.15,random_state=42)
#%%

hires_df_3= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_3,resNN_dir_Dec,saveto=None)

hires_df_3_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_3,resNN_dir_Jan,saveto=None)

data_3_Dec, labels__3_Dec=preprocess_timeseries_three_stimuli(hires_df_3)

data_3_Jan, labels__3_Jan=preprocess_timeseries_three_stimuli(hires_df_3_Jan)

data_3_final= data_3_Dec + data_3_Jan

labels_3 = labels__3_Dec + labels__3_Jan

c_group_3 = list(zip(data_3_final, labels_3))

random.shuffle(c_group_3)

data_3_final, labels_3 = zip(*c_group_3)

#split
X_train3, X_test3, y_train3, y_test3 = train_test_split(data_3_final, labels_3, test_size=0.15,random_state=42)


#%%
hires_df_4= load_bluelight_timeseries_from_drugclass(Dec_selected_meta_4,resNN_dir_Dec,saveto=None)

hires_df_4_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_meta_4,resNN_dir_Jan,saveto=None)


data_4_Dec, labels__4_Dec=preprocess_timeseries_three_stimuli(hires_df_4)

data_4_Jan, labels__4_Jan=preprocess_timeseries_three_stimuli(hires_df_4_Jan)

data_4_final= data_4_Dec + data_4_Jan

labels_4 = labels__4_Dec + labels__4_Jan

c_group4 = list(zip(data_4_final, labels_4))

random.shuffle(c_group4)

data_4_final, labels_4 = zip(*c_group4)

X_train4, X_test4, y_train4, y_test4 = train_test_split(data_4_final, labels_4, test_size=0.15,random_state=42)

#%%
hires_df_6= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_6,resNN_dir_Dec,saveto=None)

hires_df_6_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_6,resNN_dir_Jan,saveto=None)


data_6_Dec, labels__6_Dec=preprocess_timeseries_three_stimuli(hires_df_6)

data_6_Jan, labels__6_Jan=preprocess_timeseries_three_stimuli(hires_df_6_Jan)

data_6_final= data_6_Dec + data_6_Jan

labels_6 = labels__6_Dec + labels__6_Jan

c_group6 = list(zip(data_6_final, labels_6))

random.shuffle(c_group6)

data_6_final, labels_6 = zip(*c_group6)

X_train6, X_test6, y_train6, y_test6 = train_test_split(data_6_final, labels_6, test_size=0.15,random_state=42)

#%%


hires_df_7= load_bluelight_timeseries_from_drugclass(Dec_selected_meta_7,resNN_dir_Dec,saveto=None)

hires_df_7_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_meta_7,resNN_dir_Jan,saveto=None)


data_7_Dec, labels__7_Dec=preprocess_timeseries_three_stimuli(hires_df_7)

data_7_Jan, labels__7_Jan=preprocess_timeseries_three_stimuli(hires_df_7_Jan)

data_7_final= data_7_Dec + data_7_Jan

labels_7 = labels__7_Dec + labels__7_Jan

c_group7 = list(zip(data_7_final, labels_7))

random.shuffle(c_group7)

data_7_final, labels_7 = zip(*c_group7)

X_train7, X_test7, y_train7, y_test7 = train_test_split(data_7_final, labels_7, test_size=0.15,random_state=42)


#%%

hires_df_9= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_9,resNN_dir_Dec,saveto=None)

hires_df_9_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_9,resNN_dir_Jan,saveto=None)


data_9_Dec, labels_9_Dec=preprocess_timeseries_three_stimuli(hires_df_9)

data_9_Jan, labels_9_Jan=preprocess_timeseries_three_stimuli(hires_df_9_Jan)

data_9_final= data_9_Dec + data_9_Jan

labels_9 = labels_9_Dec + labels_9_Jan

c_group9 = list(zip(data_9_final, labels_9))

random.shuffle(c_group9)

data_9_final, labels_9 = zip(*c_group9)

X_train9, X_test9, y_train9, y_test9 = train_test_split(data_9_final, labels_9, test_size=0.15,random_state=42)

#%%

hires_df_10= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_10,resNN_dir_Dec,saveto=None)

hires_df_10_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_10,resNN_dir_Jan,saveto=None)


data_10_Dec, labels_10_Dec=preprocess_timeseries_three_stimuli(hires_df_10)

data_10_Jan, labels_10_Jan=preprocess_timeseries_three_stimuli(hires_df_10_Jan)

data_10_final= data_10_Dec + data_10_Jan

labels_10 = labels_10_Dec + labels_10_Jan

c_group10 = list(zip(data_10_final, labels_10))

random.shuffle(c_group10)

data_10_final, labels_10 = zip(*c_group10)

X_train10, X_test10, y_train10, y_test10 = train_test_split(data_10_final, labels_10, test_size=0.15,random_state=42)


#%%

hires_df_11= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_11,resNN_dir_Dec,saveto=None)

hires_df_11_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_11,resNN_dir_Jan,saveto=None)


data_11_Dec, labels_11_Dec=preprocess_timeseries_three_stimuli(hires_df_11)

data_11_Jan, labels_11_Jan=preprocess_timeseries_three_stimuli(hires_df_11_Jan)

data_11_final= data_11_Dec + data_11_Jan

labels_11 = labels_11_Dec + labels_11_Jan

c_group11 = list(zip(data_11_final, labels_11))

random.shuffle(c_group11)

data_11_final, labels_11 = zip(*c_group11)


X_train11, X_test11, y_train11, y_test11 = train_test_split(data_11_final, labels_11, test_size=0.15,random_state=42)

#%%
hires_df_13= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_13,resNN_dir_Dec,saveto=None)

hires_df_13_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_13,resNN_dir_Jan,saveto=None)


data_13_Dec, labels_13_Dec=preprocess_timeseries_three_stimuli(hires_df_13)

data_13_Jan, labels_13_Jan=preprocess_timeseries_three_stimuli(hires_df_13_Jan)

data_13_final= data_13_Dec + data_13_Jan

labels_13 = labels_13_Dec + labels_13_Jan

c_group13 = list(zip(data_13_final, labels_13))

random.shuffle(c_group13)

data_13_final, labels_13 = zip(*c_group13)

X_train13, X_test13, y_train13, y_test13 = train_test_split(data_13_final, labels_13, test_size=0.15,random_state=42)


#%%
hires_df_15= load_bluelight_timeseries_from_drugclass(Dec_selected_train_meta_15,resNN_dir_Dec,saveto=None)

hires_df_15_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_train_meta_15,resNN_dir_Jan,saveto=None)


data_15_Dec, labels_15_Dec=preprocess_timeseries_three_stimuli(hires_df_15)

data_15_Jan, labels_15_Jan=preprocess_timeseries_three_stimuli(hires_df_15_Jan)

data_15_final= data_15_Dec + data_15_Jan

labels_15 = labels_15_Dec + labels_15_Jan

c_group15 = list(zip(data_15_final, labels_15))

random.shuffle(c_group15)

data_15_final, labels_15 = zip(*c_group15)




hires_df_16= load_bluelight_timeseries_from_drugclass(Dec_selected_meta_16,resNN_dir_Dec,saveto=None)

hires_df_16_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_meta_16,resNN_dir_Jan,saveto=None)


data_16_Dec, labels_16_Dec=preprocess_timeseries_three_stimuli(hires_df_16)

data_16_Jan, labels_16_Jan=preprocess_timeseries_three_stimuli(hires_df_16_Jan)

data_16_final= data_16_Dec + data_16_Jan

labels_16 = labels_16_Dec + labels_16_Jan

c_group16 = list(zip(data_16_final, labels_16))

random.shuffle(c_group16)

data_16_final, labels_16 = zip(*c_group16)




hires_df_17= load_bluelight_timeseries_from_drugclass(Dec_selected_meta_17,resNN_dir_Dec,saveto=None)

hires_df_17_Jan= load_bluelight_timeseries_from_drugclass(Jan_selected_meta_17,resNN_dir_Jan,saveto=None)


data_17_Dec, labels_17_Dec=preprocess_timeseries_three_stimuli(hires_df_17)

data_17_Jan, labels_17_Jan=preprocess_timeseries_three_stimuli(hires_df_17_Jan)

data_17_final= data_17_Dec + data_17_Jan

labels_17 = labels_17_Dec + labels_17_Jan

c_group17 = list(zip(data_17_final, labels_17))

random.shuffle(c_group17)

data_17_final, labels_17 = zip(*c_group17)


X_train15, X_test15, y_train15, y_test15 = train_test_split(data_15_final, labels_15, test_size=0.15,random_state=42)
X_train16, X_test16, y_train16, y_test16 = train_test_split(data_16_final, labels_16, test_size=0.15,random_state=42)
X_train17, X_test17, y_train17, y_test17 = train_test_split(data_17_final, labels_17, test_size=0.15,random_state=42)

#%%

train_dataset = X_train2 +X_train3 + X_train4 + X_train6 + X_train7 + X_train9 + X_train10 + X_train11 + X_train13 + X_train15 +X_train16 +X_train17
df_merged= pd.concat(train_dataset, axis=0, ignore_index=True)
df_merged.reset_index(drop=True, inplace=True)
df_merged.isnull().sum()

df_train_path= os.path.join(root_dir_Dec, 'timeseries_train_dataset_12classes_8signals.csv')
df_merged.to_csv(df_train_path, index=False)


val_dataset = X_test2 +X_test3 + X_test4 + X_test6 +  X_test7 +X_test9 + X_test10 + X_test11 + X_test13 + X_test15 + X_test16 + X_test17
val_merged= pd.concat(foo_val, axis=0, ignore_index=True)
val_merged.reset_index(drop=True, inplace=True)
val_merged.isnull().sum()

df_val_path= os.path.join(root_dir_Dec, 'timeseries_val_dataset_12classes_8signals.csv')
val_merged.to_csv(df_val_path, index=False)


foo_label= y_train2+ y_train3 + y_train4 + y_train6+y_train7+ y_train9 + y_train10 + y_train11 + y_train13 + y_train15+ y_train16 +y_train17
foo_label= pd.DataFrame(foo_label) 
foo_label.reset_index(drop=True, inplace=True)
foo_label['MOA_group'].replace({ 15: 0, 16: 1, 13: 5, 17: 8 }, inplace=True)

foo_label_path= os.path.join(root_dir_Dec, 'train_labels_7classes.csv')
foo_label.to_csv(foo_label_path, index=False)


label_val= y_test2 +y_test3 + y_test4 +y_test6 +y_test7+ y_test9 + y_test10 + y_test11 + y_test13 + y_test15 +y_test16 + y_test17
label_val=pd.DataFrame(label_val)
label_val.reset_index(drop=True, inplace=True) #1954 samples
label_val['MOA_group'].replace({ 15: 0, 16: 1, 13: 5, 17: 8 }, inplace=True)

val_label_path= os.path.join(root_dir_Dec, 'val_labels_7classes.csv')
label_val.to_csv(val_label_path, index=False)














