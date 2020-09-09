#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 10:54:05 2020

@author: liuziwei
"""

import tables
import time
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from helper import make_well_id,filter_timeseries, find_motion_changes, plot_stimuli
from tierpsytools_read_data_get_timeseries import read_timeseries

from helper import (find_motion_changes,
                    read_metadata,
                    plot_stimuli,
                    load_bluelight_timeseries_from_results,
                    just_load_one_timeseries, count_motion_modes,
                    get_frac_motion_modes,
                    HIRES_COLS)
import os
from helper import downsample_timeseries
#%% Where files are
root_dir = Path('/Volumes/Ashur Pro2/SyngentaScreen')
resNN_dir = root_dir / 'Results_NN'

figures_dir = root_dir / 'Figures_rerun'
figures_dir.mkdir(exist_ok=True)

meta_updated=pd.read_csv('/Volumes/Ashur Pro2/SyngentaScreen/metadata_alldrugs_N2.csv')
#%% Get videos for DMSO
def select_meta(df, filter_dic):
    df_=df.copy()
    for k, v in filter_dic.items():
        df_=df_[df_[k]==v]
        
    return df_

selected_meta_DMSO=select_meta(meta_updated, {'drug_type':'DMSO'})

selected_meta_NoComp=select_meta(meta_updated, {'drug_type':'NoCompound'})
#%%

ID_COLS = ['date_yyyymmdd','imaging_plate_id','well_name']

CATEG_COLS = ID_COLS + ['worm_strain']


COLS_MOTION_CHANGE = ['fw2bw', 'bw2fw',  # diff is 2 or -2
                      'bw2st', 'st2fw',  # diff is 1, motion 0 or 1
                      'fw2st', 'st2bw']  # diff is -1, motion 0 or -1

COLS_MOTION_MODES = ['is_fw', 'is_bw', 'is_st', 'is_nan']

HIRES_COLS = ['worm_index', 'timestamp', 'speed', 'length','speed_neck','speed_head_base','speed_head_tip','curvature_midbody', 
              'motion_mode','speed_hips','speed_tail_base','speed_tail_tip']

def load_bluelight_timeseries_from_one_compound(
        metadata_df,results_dir,saveto=None):

    hires_df = []  # this will only have a few features
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
        data_g = data.groupby('well_name')
        

        data = pd.merge(data, md_group[CATEG_COLS], how='left', on='well_name')
        #data['date_yyyymmdd'] = data['date_yyyymmdd'].astype(str)
        
        hires_data = data[CATEG_COLS + HIRES_COLS].copy()
    
        # create a single unique well id
        hires_data['well_id'] = make_well_id(hires_data)

        # find motion changes
        #find_motion_changes(hires_data)
        # import pdb; pdb.set_trace()
        # set some columns to categorical for memory saving
        for col in CATEG_COLS:
            hires_data[col] = hires_data[col].astype(cat_types[col])

        if saveto is None:
            # now grow list
            hires_df.append(hires_data)
            
        else:
            # save to disk
            is_append = False if (gcounter == 0) else True
            data.to_hdf(
                saveto, 'timeseries_df', format='table', append=is_append)
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
        # out
        return  hires_df
#%%
hires_df_NoComp = load_bluelight_timeseries_from_one_compound(selected_meta_NoComp,resNN_dir)

  
plt.close('all')
plt.tight_layout()

feats_toplot = ['speed','length','speed_neck','speed_head_base','speed_head_tip','curvature_midbody','speed_hips','speed_tail_base','speed_tail_tip']

with PdfPages(figures_dir / 'N2_no_compound_selected_feature.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        sns.lineplot(x='timestamp', y=feat,
                     hue='motion_mode', # stationary mode removed, forward mode=1, backward mode =-1
                     style='motion_mode', 
                     palette='Set2',#'worm_strain',
                     data= hires_df_NoComp.query('motion_mode != 0'),
                     estimator=np.mean, ci='sd',
                     legend='full')
        plt.xlim(xmin = 0)
        plt.legend(title='Motion_mode', loc='lower right', labels=['backward', 'forward'],prop=fontP)
        plot_stimuli(ax=ax, units='frames')
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

#%%

hires_df_NoComp = hires_df_NoComp.rename({'speed':'velocity'}, axis=1)
hires_df_NoComp['speed'] = hires_df_NoComp['velocity'].abs()

hires_df_NoComp['curvature_midbody_abs'] = hires_df_NoComp['curvature_midbody'].abs()
hires_df_NoComp['d_curvature_midbody_abs'] = hires_df_NoComp['d_curvature_midbody'].abs()

fig, ax = plt.subplots()
sns.lineplot(x='timestamp', y='speed', palette='Set1', data=hires_df_NoComp,estimator=np.mean, ci='sd',legend=False)
plt.title('Average curvature_midbody')
plt.xlim(xmin = 0)
plot_stimuli(ax=ax, units='frames')
labels = [hires_df_DMSO['timestamp'] / fps]
ax.set_xticklabels(labels)

fig, ax = plt.subplots()
sns.lineplot(x='timestamp', y='curvature_midbody_abs', palette='Set1', data=hires_df_NoComp,estimator=np.mean, ci='sd',legend=False)
plt.title('Average curvature_midbody')
plt.xlim(xmin = 0)
plot_stimuli(ax=ax, units='frames')
labels = [hires_df_DMSO['timestamp'] / fps]
ax.set_xticklabels(labels)


fig, ax = plt.subplots()
sns.lineplot(x='timestamp', y='d_curvature_midbody_abs', palette='Set1', data=hires_df_NoComp,estimator=np.mean, ci='sd',legend=False)
plt.title('Average d_curvature_midbody')
plt.xlim(xmin = 0)
plot_stimuli(ax=ax, units='frames')

fig, ax = plt.subplots()
sns.lineplot(x='timestamp', y='curvature_midbody', hue='motion_mode', palette='Set1', data=hires_df_NoComp.query('motion_mode != 0'),estimator=np.mean, ci='sd',legend='full')
plt.title('Average curvature_midbody DMSO,two_motion_modes')
plt.legend(title='Motion_mode', loc='lower right', labels=['backward','forward'])
plt.xlim(xmin = 0)
plot_stimuli(ax=ax, units='frames')

#%% how to define reversal

fig, ax = plt.subplots()
hires_df_NoComp.plot(x='timestamp', y='motion_up', ax=ax)
hires_df_NoComp.plot(x='timestamp', y='motion_down', ax=ax)
plot_stimuli(ax=ax, units='frames')
ax.set_ylabel('event counts')

bin_width_s = 5
fps = 25
binedges = np.arange(0, 380*fps, step=bin_width_s*fps)
alpha = 0.5

fig, ax = plt.subplots()
hires_df_NoComp.query('st2fw == True')['timestamp'].hist(bins=binedges,label='st2fw',alpha=alpha)
hires_df_NoComp.query('st2bw == True')['timestamp'].hist(bins=binedges,label='st2bw',alpha=alpha)
ax.set_ylabel('event counts')
ax.set_xlabel('timestamp')
ax.legend()
plot_stimuli(ax=ax, units='frames')


fig, ax= plt.subplots()
sns.lineplot(x='time_s', y='relative_to_body_angular_velocity_neck_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',palette = "Set2",ax=ax)
sns.lineplot(x='time_s', y='relative_to_body_angular_velocity_hips_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',palette = "Set2",ax=ax)
fig.legend(labels=['relative_to_body_angular_velocity_neck_abs','relative_to_body_angular_velocity_hips_abs'],loc="upper right")
plot_stimuli(ax=ax, units='s⁻¹')
ax.set_ylabel('Angular_velocity_abs')
fig.tight_layout()

#%% NoComp graph

hires_df_NoComp['time_s'] = hires_df_NoComp['timestamp'] / 25

hires_df_NoComp['unique_id'] = 0    
df_3= hires_df.groupby(by=['date_yyyymmdd','imaging_plate_id','well_name', 'worm_index'])['unique_id'].transform(lambda x: len(x)>1)
hires_df.loc[df_3,'unique_id'] = hires_df.loc[df_3,['date_yyyymmdd','imaging_plate_id','well_name', 'worm_index']].astype(str).sum(1).factorize()[0] + 1


fig, ax = plt.subplots()
#colors = ["crimson","deepskyblue","peru"]
#colors = ["seagreen","deepskyblue","peru"]
#colors = ["darkorange","deepskyblue","peru"]
colors = ["dimgray","deepskyblue","peru"]
#colors= [ "#3498db","#9b59b6", "#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(colors))
sns.color_palette(colors)
sns.lineplot(x='time_s', y='speed_head_tip_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',ax=ax)
sns.lineplot(x='time_s', y='speed_head_base_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',ax=ax)
sns.lineplot(x='time_s', y='speed_neck_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',ax=ax)
sns.lineplot(x='time_s', y='speed_hips_abs',data=hires_df_NoComp,estimator=np.mean, ci='sd',ax=ax)
ax.legend(labels=['speed_head_tip','speed_head_base','speed_neck','speed_hips'],loc="lower right", ncol=2)
plot_stimuli(ax=ax, units='s')
ax.set_ylabel('Speed (μm/ss)')
ax.set_xlabel('time (s)')
plt.xlim(xmin = 0, xmax=360)


fig, ax = plt.subplots()
colors= [ "palevioletred","darkcyan"]
sns.set_palette(sns.color_palette(colors))
sns.color_palette(colors)
sns.lineplot(x='time_s', y='curvature_midbody_abs', data=hires_df_NoComp,estimator=np.mean, ci='sd',legend='full')
ax.set_ylabel('angular velocity of neck relative to body_abs (rad/s)')
ax.set_xlabel('time (s)')
plt.legend(loc='lower right', labels=['ang_velocity_neck_rel_to_body_abs'])
plt.xlim(xmin = 0, xmax=360)
plot_stimuli(ax=ax, units='s')



fig, ax = plt.subplots()
colors= [ "mediumpurple"]
#sns.set_palette(sns.color_palette(colors))
sns.color_palette(colors)
sns.lineplot(x='time_s', y='angular_velocity_head_base_abs', data=hires_df_NoComp,estimator=np.mean, ci='sd',legend='full')
ax.set_ylabel('angular velocity of head base_abs (rad/s)')
ax.set_xlabel('time (s)')
plt.legend(loc='lower right', labels=['ang_velocity_head_base_abs'])
plt.xlim(xmin = 0, xmax=360)
plot_stimuli(ax=ax, units='s')




fig, ax = plt.subplots()
colors = ["darkorange", "seagreen"]
customPalette=sns.color_palette(colors)
ax=sns.lineplot(x='time_s', y='velocity', hue='motion_mode', data=hires_df_NoComp.query('motion_mode != 0'),estimator=np.mean, ci='sd',palette=customPalette,legend='full')
plt.title('Population mean velocity in two motion modes')
plt.legend(title='Motion_mode', loc='lower right', labels=['backward','forward'])
plt.xlim(xmin = 0, xmax=360)
ax.set_ylabel('Velocity (μm/s)')
ax.set_xlabel('time (s)')
plot_stimuli(ax=ax, units='s')
fig.tight_layout(pad=0, h_pad=0, w_pad=0)


#%%Compare DMSO and NoComp

fig, ax = plt.subplots()
ax.plot(hires_df_NoComp.timestamp, hires_df_NoComp.speed, label='No Compound', color='blue')
ax.plot(hires_df_DMSO.timestamp, hires_df_DMSO.speed, label='DMSO', color='magenta')
ax.legend()
plt.title('Average speed NoComp compared to DMSO')
plt.xlabel('timestamp')
plt.ylabel('speed')
plot_stimuli(ax=ax, units='frames')
plt.xlim(xmin = 0)

 #%%

#Compare N2 DMSO vs a drug
Serotonin_antagonist = ['CSAA026102','CSAA149470', 'CSAA189597','CSAA532845','CSAA871813'] #class2

Serotonin_agonist = ['CSAA145723','CSAA161867','CSAA173130'] #class1
                       

AchE_inhibitors_Carbamate=['CSAA012029', 'CSAA008422', 'CSAA020833', 'CSAA011840', 'CSAA012836', 
                           'CSAA044014', 'CSAA016712', 'CSAA035078', 'CSAA000937'] #class3


AchE_inhibitors_Organophosphate= ['CSAA021296', 'CSAA128717', 'CSAA005330', 'CSAA006770', 'CSAA017604'] #class4

GABA_antagonist = ['CSCA109859', 'CSAA123410', 'CSCC812077', 'CSCA136013', 'CSCC811993', 'CSAA632702', 'CSAA398011', 'CSAA159511'] #class6

GluCl_agonist= ['CSCD010279', 'CSAA632597', 'CSCC201954', 'CSAA466656'] #class7


spiroindolines = ['CSCC222657', 'CSCC232026', 'CSAC270548', 'CSCC218769', 'CSCC230937', 'CSCD068947', 
                  'CSCC223039', 'CSCC202642', 'CSCD067122']#vAChT inhibitor #class17


all_selected_drugs = spiroindolines + Serotonin_antagonist + Serotonin_agonist 


for drug in tqdm(all_selected_drugs):
    if drug in Serotonin_agonist:
        globals()['selected_meta_%s' % drug ]=select_meta(Dec_meta_class1, {'drug_type':drug})
        globals()['meta_%s' % drug ]= globals()['selected_meta_%s' % drug ].groupby('imaging_plate_drug_concentration')
        globals()['hires_df_%s' % drug]= pd.DataFrame()
        for k, concentration in tqdm(globals()['meta_%s' % drug ]):
            df = load_bluelight_timeseries_from_one_compound(concentration,resNN_dir_Dec) 
            df['drug_concentration'] = pd.Series(concentration['imaging_plate_drug_concentration'].unique().tolist() * len(df.index))
            globals()['hires_df_%s' % drug]= globals()['hires_df_%s' % drug].append(df)

                     
for drug in tqdm(Serotonin_antagonist): 
    globals()['hires_df_%s' % drug] = globals()['hires_df_%s' % drug].rename({'speed':'velocity'}, axis=1)
    globals()['hires_df_%s' % drug]['speed'] = globals()['hires_df_%s' % drug]['velocity'].abs()
    globals()['hires_df_%s' % drug]['curvature_midbody_abs'] = globals()['hires_df_%s' % drug]['curvature_midbody'].abs()
    globals()['hires_df_%s' % drug]['d_curvature_midbody_abs'] = globals()['hires_df_%s' % drug]['d_curvature_midbody'].abs()


feats = ['speed','speed_neck']

for drug in tqdm(Serotonin_agonist): 
    for feat in tqdm(feats):
        with PdfPages(resNN_dir_Dec / 'N2worm_{}_DMSOvs{}.pdf'.format(feat,drug), keep_empty=False) as pdf:
            fig, ax = plt.subplots()
            for k, concentration in globals()['hires_df_%s' % drug].groupby('drug_concentration'):
                sns.lineplot(x='timestamp', y=feat,
                             palette='blue',
                             data=concentration,
                             estimator=np.mean, ci='sd',
                             legend=False)
                sns.lineplot(x='timestamp', y=feat,  palette='green', 
                             data=hires_df_DMSO, estimator=np.mean, ci='sd',legend= False)           
            plt.xlim(xmin = 0)
            plt.legend(title='Compound given', loc='upper right', labels=['{}uM'.format(tuple(concentration['drug_concentration'].unique()), '0.1%DMSO'],prop=fontP)
            plot_stimuli(ax=ax, units='frames')
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

#%% Heatmap for reversals


df_test=hires_df_NoComp[['unique_id','time_s','motion_mode']]

df_test =  df_test.groupby('unique_id').filter(lambda g: len(g) >3000)

# Filling the missing timestamps for the worm tracking for parts of the video

piv_dropna= pd.pivot_table(df_test, values="motion_mode",index=["unique_id"], columns=["time_s"])
piv_dropna.columns = piv_dropna.columns.astype(str).str.split('_', expand=True)
piv_dropna.drop('360.04', axis=1, inplace=True)
piv_dropna.drop('360.08', axis=1, inplace=True)
piv_cols=list(piv.columns)

piv_dropna.interpolate(method ='linear', axis=1, limit=625,inplace=True)
piv_dropna.dropna(axis=0, how='any',inplace=True) #200 worms left

piv_dropna.fillna(-2, inplace=True)
piv_dropna.reset_index(inplace=True) 
piv_dropna.drop(columns=['unique_id'],inplace=True)#649 worms

arr=np.asarray(piv_dropna)

plt.imshow(arr)  
plt.colorbar()
plt.xlabel('Timestamp')
plt.ylabel('Individual')



xticks = np.linspace(0, 9000, 25, dtype=np.int)
depth_list=piv_dropna.columns.values

xticklabels = [depth_list[idx] for idx in xticks]
#xticklabels=[round(num) for num in xticklabels]

yticks = np.linspace(0, 693, 30, dtype=np.int)
index_list=piv_dropna.index.values.tolist()
yticklabels = [index_list[idx] for idx in yticks]

fig, ax = plt.subplots()
ax = sns.clustermap(piv_dropna, method='single',col_cluster=False,cmap="mako")
ax.ax_heatmap.set_xlabel('Time (s)')
ax.ax_heatmap.set_ylabel('Individuals')
ax.ax_heatmap.set_title('Single-linkage hierarchy clustering for single individual worms in different motion modes')
ax.ax_heatmap.set_yticks(yticks)
ax.ax_heatmap.set_yticklabels(yticklabels)
ax.ax_heatmap.set_xticks(xticks)
ax.ax_heatmap.set_xticklabels(xticklabels)
ax.cax.remove()

#custom paleete
cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=30, as_cmap=True)


fig, axs = plt.subplots(3, sharex=True)
axs[0].plot((arr== -1).sum(axis=0) / (arr != -2).sum(axis=0))

axs[0].set_ylabel('Frac_backward')

axs[1].plot((arr == 0).sum(axis=0) / (arr != -2).sum(axis=0))

axs[1].set_ylabel('Frac_ at pause')

axs[2].plot((arr == 1).sum(axis=0) / (arr != -2).sum(axis=0))
axs[2].set_ylabel('Frac_forward')
axs[2].set_xlabel('Timestamps')


fig, ax = plt.subplots()
plt.plot(y_forwards)
plt.plot(y_forwards + y_stationary)  # sum to stack them in the plot
plt.plot(y_forwards + y_stationary + y_backwards)








                          
                          
                          