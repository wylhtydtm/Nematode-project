#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:34:07 2020

@author: liuziwei
"""


import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tierpsytools_read_data_get_timeseries import read_timeseries

from helper import (read_metadata,make_well_id,filter_timeseries,plot_stimuli,
                    load_bluelight_timeseries_from_results,
                    just_load_one_timeseries, count_motion_modes,
                    get_frac_motion_modes,
                    HIRES_COLS,select_meta)

#%%
root_dir_Dec = Path('/Volumes/Ashur Pro2/SyngentaScreen')
resNN_dir_Dec = root_dir_Dec / 'Results_NN'
metadata_JanFeb= pd.read_csv('/Volumes/Ashur Pro2/SyngentaStrainScreen/metadata_JanFeb_N2.csv')
resNN_dir_Jan= Path('/Volumes/Ashur Pro2/SyngentaStrainScreen/Results_NN')
metadata_Dec= pd.read_csv(root_dir_Dec /'metadata_alldrugs_N2.csv')
fig_dif=Path('/Users/liuziwei/Desktop/fig_drug')
#meta= pd.concat([metadata_Dec,metadata_JanFeb])

drug_class=list(set(metadata_Dec['MOA_group']))

for drug in drug_class:
   globals()['Dec_meta_class%s' % int(drug)]= select_meta(metadata_Dec,{'MOA_group':drug})
   globals()['Dec_meta_class%s' % int(drug)].reset_index(drop= True, inplace=True)
   globals()['Dec_meta_class%s' % int(drug)].drop(columns=['Unnamed: 5'], inplace=True)


for drug in drug_class:
   globals()['Jan_meta_class%s' % int(drug)]= select_meta(metadata_JanFeb,{'MOA_group':drug})
   globals()['Jan_meta_class%s' % int(drug)].reset_index(drop= True, inplace=True)
   globals()['Jan_meta_class%s' % int(drug)].drop(columns=['Unnamed: 5'], inplace=True)

meta_NoComp=select_meta(metadata_Dec, {'drug_type':'NoCompound'})

#%%
ID_COLS = ['well_name','imaging_plate_id','date_yyyymmdd']

CATEG_COLS = ID_COLS + ['worm_strain']


COLS_MOTION_CHANGE = ['fw2bw', 'bw2fw',  # diff is 2 or -2
                      'bw2st', 'st2fw',  # diff is 1, motion 0 or 1
                      'fw2st', 'st2bw']  # diff is -1, motion 0 or -1

COLS_MOTION_MODES = ['is_fw', 'is_bw', 'is_st', 'is_nan']

HIRES_COLS = ['worm_index', 'timestamp','length','speed','angular_velocity_head_base','curvature_midbody','angular_velocity_tail_base','motion_mode']

SAVED_COlS=['timestamp','well_id','length','speed','curvature_midbody','angular_velocity_head_base','angular_velocity_tail_base','motion_mode','CSN','imaging_plate_drug_concentration','MOA_group','unique_id','worm_index']
            
                
def load_bluelight_timeseries_from_drugclass(
        metadata_df,results_dir,saveto=None):

    hires_df = []

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
        data = pd.merge(data, md_group[CATEG_COLS], how='left', on='well_name')
        data['date_yyyymmdd'] = data['date_yyyymmdd'].astype(str)
        hires_data = data[CATEG_COLS + HIRES_COLS].copy()

          # create a single unique well id
        hires_data['well_id'] = make_well_id(hires_data)
        # set some columns to categorical for memory saving
        for col in CATEG_COLS:
            hires_data[col] = hires_data[col].astype(cat_types[col])

        if saveto is None:
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
hires_df_NoComp= load_bluelight_timeseries_from_drugclass(meta_NoComp,resNN_dir_Dec,saveto=None)

hires_df_NoComp = hires_df_NoComp.rename({'speed':'velocity'}, axis=1)
hires_df_NoComp['speed (μm/s)'] = hires_df_NoComp['velocity'].abs()
hires_df_NoComp['curvature_midbody_abs (rad/μm)'] = hires_df_NoComp['curvature_midbody'].abs() 

#%%
hires_df_DMSO= load_bluelight_timeseries_from_drugclass(Dec_meta_class0,resNN_dir_Dec,saveto=None)
hires_df_DMSO = hires_df_DMSO.rename({'speed':'velocity'}, axis=1)
hires_df_DMSO['speed(μm/s)'] = hires_df_DMSO['velocity'].abs()
hires_df_DMSO['curvature_midbody_abs (rad/μm)'] = hires_df_DMSO['curvature_midbody'].abs() 
hires_df_DMSO['time_s'] = hires_df_DMSO['timestamp'] / 25
hires_df_DMSO['angular_velocity_head_base(rad/μm)'] = hires_df_DMSO['angular_velocity_head_base'].abs() 
hires_df_DMSO['angular_velocity_tail_base(rad/μm)'] = hires_df_DMSO['angular_velocity_tail_base'].abs() 

#%% Compare DMSO and NoCompound

fig, ax = plt.subplots()
sns.color_palette("Set2")
sns.lineplot(x='time_s', y='speed',data=hires_df_NoComp,estimator=np.mean, ci='sd',ax=ax)
sns.lineplot(x='time_s', y='speed',data=hires_df_DMSO,estimator=np.mean, ci='sd',ax=ax)
ax.legend(labels=['NoComp','1% DMSO'],loc="lower right")
plot_stimuli(ax=ax, units='s')
ax.set_ylabel('Speed (μm/s)')
ax.set_xlabel('time (s)')
plt.xlim(xmin = 0, xmax=360)


#%%
#Serotonin anonist and antagonist
hires_df_class1=load_bluelight_timeseries_from_drugclass(Dec_meta_class1,resNN_dir_Dec,saveto=None)
hires_df_class_1=load_bluelight_timeseries_from_drugclass(Jan_meta_class1,resNN_dir_Jan,saveto=None)
hires_df_class2=load_bluelight_timeseries_from_drugclass(Dec_meta_class2,resNN_dir_Dec,saveto=None)
hires_df_class_2=load_bluelight_timeseries_from_drugclass(Jan_meta_class2,resNN_dir_Jan,saveto=None)

hires_df_class1_final = pd.concat([hires_df_class1,hires_df_class_1])
hires_df_class1_final= hires_df_class1_final.rename({'speed':'velocity'}, axis=1)
hires_df_class1_final['speed(μm/s)'] = hires_df_class1_final['velocity'].abs()
hires_df_class1_final['curvature_midbody_abs(rad/μm)'] = hires_df_class1_final['curvature_midbody'].abs()
hires_df_class1_final['time_s'] = hires_df_class1_final['timestamp'] / 25


#
hires_df_class2_final = pd.concat([hires_df_class2,hires_df_class_2])
hires_df_class2_final= hires_df_class2_final.rename({'speed':'velocity'}, axis=1)
hires_df_class2_final['speed(μm/s)'] = hires_df_class2_final['velocity'].abs()
hires_df_class2_final['curvature_midbody_abs(rad/μm) '] = hires_df_class2_final['curvature_midbody'].abs()


#Filter out the worms being tracked for less than 1500
hires_df_class1_final = hires_df_class1_final.groupby('unique_id').filter(lambda g: len(g) >1500)
hires_df_class2_final= hires_df_class2_final.groupby('unique_id').filter(lambda g: len(g) >1500)



big_data=[]

for k, df in hires_df_class2_final.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
    print(df['unique_id'].nunique())
    random_selected_ids = np.random.choice(df['unique_id'].unique(), 10, replace=False)
    df_random_selected = df[df['unique_id'].isin(random_selected_ids)]
    big_data.append(df_random_selected)


hires_dfclass2=pd.concat(big_data,axis=0,ignore_index=True)
hires_dfclass2['time(s)'] = hires_dfclass2['timestamp'] / 25


#Testing moving average
hires_df_class1_final['MV_speed(μm/s)'] = hires_df_class1_final.groupby('unique_id')['speed(μm/s)'].transform(lambda x: x.rolling(25, 1).mean())


feats=['speed(μm/s)']

for feat in tqdm(feats):
    with PdfPages(fig_dif / 'N2_wifDMSO_group2_compoundnamespeed.pdf', keep_empty=False) as pdf:
        for k, df in foo.groupby(by=['Compound name', 'imaging_plate_drug_concentration']):
            fig, ax = plt.subplots()
            sns.lineplot(x='time(s)', y='speed(μm/s)', palette='blue',data=df,estimator=np.mean, ci='sd',legend=False)
            sns.lineplot(x='time_s', y='speed(μm/s)',  palette='green', data=hires_df_DMSO, estimator=np.mean, ci='sd',legend= False)           
            plt.xlim(xmin = 0, xmax=360)
            ax.set_ylabel('speed (μm/s)')
            ax.set_xlabel('time (s)')
            plt.legend(title='Compound given', loc='upper right', labels=['{}:{}uM'.format(k[0],k[1]), '0.1%DMSO'])
            plot_stimuli(ax=ax, units='s')
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


#%%

hires_df_class9=load_bluelight_timeseries_from_drugclass(Dec_meta_class9,resNN_dir_Dec,saveto=None)
hires_df_class_9=load_bluelight_timeseries_from_drugclass(Jan_meta_class9,resNN_dir_Jan,saveto=None)
hires_df_class23=load_bluelight_timeseries_from_drugclass(Dec_meta_class23,resNN_dir_Dec,saveto=None)
hires_df_class_23=load_bluelight_timeseries_from_drugclass(Jan_meta_class23,resNN_dir_Jan,saveto=None)

hires_df_class9_final = pd.concat([hires_df_class9,hires_df_class_9])
hires_df_class9_final= hires_df_class9_final.rename({'speed':'velocity'}, axis=1)
hires_df_class9_final['speed(μm/s)'] = hires_df_class9_final['velocity'].abs()
hires_df_class9_final['curvature_midbody_abs(rad/μm)'] = hires_df_class9_final['curvature_midbody'].abs()
  

hires_df_class23_final = pd.concat([hires_df_class23,hires_df_class_23])
hires_df_class23_final= hires_df_class23_final.rename({'speed':'velocity'}, axis=1)
hires_df_class23_final['speed(μm/s)'] = hires_df_class23_final['velocity'].abs()
hires_df_class23_final['curvature_midbody_abs(rad/μm)'] = hires_df_class23_final['curvature_midbody'].abs()



hires_df_class9_final = hires_df_class9_final.groupby('unique_id').filter(lambda g: len(g) >1500)
hires_df_class23_final= hires_df_class23_final.groupby('unique_id').filter(lambda g: len(g) >1500)


big_data_9=[]

for k, df in hires_df_class9_final.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
    try:
        random_selected_ids = np.random.choice(df['unique_id'].unique(), 10, replace=False)
        df_random_selected = df[df['unique_id'].isin(random_selected_ids)]
        big_data_9.append(df_random_selected)
    except ValueError:
        continue

hires_dfclass9=pd.concat(big_data_9,axis=0,ignore_index=True)
hires_dfclass9['time(s)'] = hires_dfclass9['timestamp'] / 25



hires_dfclass23=pd.concat(big_data_23,axis=0,ignore_index=True)
hires_dfclass23['time(s)'] = hires_dfclass23['timestamp'] / 25



fig_dif=Path('/Users/liuziwei/Desktop/fig_drug')

feats=['speed(μm/s)']

for feat in tqdm(feats):
    with PdfPages(fig_dif / 'N2_wifDMSO_group4_10_worms_speed.pdf', keep_empty=False) as pdf:
        for k, df in hires_dfclass4.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
            fig, ax = plt.subplots()
            sns.lineplot(x='time(s)', y=feat, palette='blue',data=df,estimator=np.mean, ci='sd',legend=False)
            sns.lineplot(x='time_s', y=feat,  palette='green', data=hires_df_DMSO, estimator=np.mean, ci='sd',legend= False)           
            plt.xlim(xmin = 0, xmax=360)
            ax.set_xlabel('time (s)')
            plt.legend(title='Compound given', loc='upper right', labels=['{}:{}uM'.format(k[0],k[1]), '0.1%DMSO'])
            plot_stimuli(ax=ax, units='s')
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


#%%
hires_df_class4=load_bluelight_timeseries_from_drugclass(Dec_meta_class4,resNN_dir_Dec,saveto=None)
hires_df_class_4=load_bluelight_timeseries_from_drugclass(Jan_meta_class4,resNN_dir_Jan,saveto=None)


hires_df_class4_final = pd.concat([hires_df_class4,hires_df_class_4])
hires_df_class4_final= hires_df_class4_final.rename({'speed':'velocity'}, axis=1)
hires_df_class4_final['speed(μm/s)'] = hires_df_class4_final['velocity'].abs()
hires_df_class4_final['curvature_midbody_abs(rad/μm)'] = hires_df_class4_final['curvature_midbody'].abs()
hires_df_class4_final['angular_velocity_head_base(rad/μm)'] = hires_df_class4_final['angular_velocity_head_base'].abs() 
hires_df_class4_final['angular_velocity_tail_base(rad/μm)'] = hires_df_class4_final['angular_velocity_tail_base'].abs() 



hires_df_class4_final = hires_df_class4_final.groupby('unique_id').filter(lambda g: len(g) >1500)

big_data_4=[]

for k, df in hires_df_class4_final.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
    try:
        random_selected_ids = np.random.choice(df['unique_id'].unique(), 10, replace=False)
        df_random_selected = df[df['unique_id'].isin(random_selected_ids)]
        big_data_4.append(df_random_selected)
    except ValueError:
        continue
    

hires_dfclass4=pd.concat(big_data_4,axis=0,ignore_index=True)
hires_dfclass4['time(s)'] = hires_dfclass4['timestamp'] / 25

hires_4=pd.merge(hires_dfclass4, drug_name, left_on='CSN', right_on='CSN')


with PdfPages(fig_dif / 'N2_wifDMSO_group4_compoundnamelength.pdf', keep_empty=False) as pdf:
    for k, df in hires_4.groupby(by=['Compound name', 'imaging_plate_drug_concentration']):
        fig, ax = plt.subplots()
        sns.lineplot(x='time(s)', y='length', palette='blue',data=df,estimator=np.mean, ci='sd',legend=False)
        sns.lineplot(x='time_s', y='length',  palette='green', data=hires_df_DMSO, estimator=np.mean, ci='sd',legend= False)           
        plt.xlim(xmin = 0, xmax=360)
        ax.set_xlabel('time (s)')
        plt.legend(title='Compound given', loc='upper right', labels=['{}:{}uM'.format(k[0],k[1]), '0.1%DMSO'])
        plot_stimuli(ax=ax, units='s')
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


#%%
hires_df_class17=load_bluelight_timeseries_from_drugclass(Dec_meta_class17,resNN_dir_Dec,saveto=None)
hires_df_class_17=load_bluelight_timeseries_from_drugclass(Jan_meta_class17,resNN_dir_Jan,saveto=None)


hires_df_class17_final = pd.concat([hires_df_class17,hires_df_class_17])
hires_df_class17_final= hires_df_class17_final.rename({'speed':'velocity'}, axis=1)
hires_df_class17_final['speed(μm/s)'] = hires_df_class17_final['velocity'].abs()
hires_df_class17_final['curvature_midbody_abs(rad/μm)'] = hires_df_class17_final['curvature_midbody'].abs()

hires_df_class17_final = hires_df_class17_final.groupby('unique_id').filter(lambda g: len(g) >1500)

hires_df_class17_final['angular_velocity_head_base(rad/μm)'] = hires_df_class17_final['angular_velocity_head_base'].abs() 
hires_df_class17_final['angular_velocity_tail_base(rad/μm)'] = hires_df_class17_final['angular_velocity_tail_base'].abs() 



big_data_17=[]

for k, df in hires_df_class17_final.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
    try:
        random_selected_ids = np.random.choice(df['unique_id'].unique(), 10, replace=False)
        df_random_selected = df[df['unique_id'].isin(random_selected_ids)]
        big_data_17.append(df_random_selected)
    except ValueError:
        continue
    

hires_dfclass17=pd.concat(big_data_17,axis=0,ignore_index=True)
hires_dfclass17['time(s)'] = hires_dfclass17['timestamp'] / 25


hires_17=pd.merge(hires_dfclass17, drug_name, left_on='CSN', right_on='CSN')


with PdfPages(fig_dif / 'N2_wifDMSO_group17_compoundsheadbase.pdf', keep_empty=False) as pdf:
    for k, df in hires_dfclass17.groupby(by=['CSN', 'imaging_plate_drug_concentration']):
        fig, ax = plt.subplots()
        sns.lineplot(x='time(s)', y='angular_velocity_head_base(rad/μm)', palette='blue',data=df,estimator=np.mean, ci='sd',legend=False)
        sns.lineplot(x='time_s', y='angular_velocity_head_base(rad/μm)',  palette='green', data=hires_df_DMSO, estimator=np.mean, ci='sd',legend= False)           
        plt.xlim(xmin = 0, xmax=360)
        ax.set_xlabel('time (s)')
        plt.legend(title='Compound given', loc='upper right', labels=['{}:{}uM'.format(k[0],k[1]), '0.1%DMSO'])
        plot_stimuli(ax=ax, units='s')
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)




















