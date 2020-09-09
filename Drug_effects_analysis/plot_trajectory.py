#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot trajectory using Tierpsy tools，and ploting trajectory, speed, curvature_midbody_for a representative worm
@author: ziwei
"""


import matplotlib.pyplot as plt
import pandas as pd
import pdb
from plot_worm_trajectories import _plot_trajectory，plot_well_trajectories，plot_multiwell_trajectories
import seaborn as sns

#%%

file = "/Volumes/Ashur Pro2/SyngentaScreen/Results_NN/" + \
    "20191206/syngenta_screen_run1_bluelight_20191206_144444.22956805/" + \
        "metadata_featuresN.hdf5"

well_name = pd.read_hdf(file, key='timeseries_data', mode='r')['well_name']

df = pd.read_hdf(file, key='trajectories_data', mode='r')

df.insert(0, 'well_name', well_name)

print(df.shape)
df = df[df['was_skeletonized'].astype(bool)]
print(df.shape)
df = df[df['well_name']!='n/a']
print(df.shape)

for wormid in df['worm_index_joined'].unique()[1]:
    _plot_trajectory(
        df.loc[df['worm_index_joined']== 2, ['coord_x', 'coord_y']],
        subsampling_rate=2,
        xlim=[df['coord_x'].min()*1.05, df['coord_x'].max()*1.05],
        ylim=[df['coord_y'].min()*1.05, df['coord_y'].max()*1.05])

for well in well_name.unique():
    n_traj = df.loc[df['well_name']==well, 'worm_index_joined'].nunique()
    plot_well_trajectories(
        df.loc[df['well_name']==well, ['coord_x', 'coord_y']],
        df.loc[df['well_name']==well, 'worm_index_joined'],
        subsampling_rate=1, title='well {} : {} trajectories'.format(well, n_traj),
        xlim=[df.loc[df['well_name']==well, 'coord_x'].min()*0.95,
                   df.loc[df['well_name']==well, 'coord_x'].max()*1.05],
         ylim=[df.loc[df['well_name']==well, 'coord_y'].min()*0.95,
                   df.loc[df['well_name']==well, 'coord_y'].max()*1.05])

plot_multiwell_trajectories(
    df[['coord_x', 'coord_y']], df['well_name'], df['worm_index_joined'],
    wells_to_plot = ['H11', 'H9', 'H10', 'G10', 'G9', 'G11','G12','F11','F9','F12','F10','E10',
                         'E11','E12','E9','H12'],
        subsampling_rate=25
        )
    
    
df = df.query('well_name =="G5"')
df_new=  df.query('0< timestamp_raw < 1500')
df_new_blue=df.query('1500< timestamp_raw < 1750')
df_new_post=df.query('1750< timestamp_raw < 4000')

_plot_trajectory(
    df_new.loc[df_new['well_name']=='G5', ['coord_x', 'coord_y']],
    subsampling_rate=25,xlim=[520, 600],ylim=[1500, 1800])


fig, axes = plt.subplots()
xycoord = df_new.loc[df_new['well_name']=='G5', ['coord_x', 'coord_y']]#0-1500
xycoord = xycoord.iloc[::25, :] #0-1500
axes.plot(*xycoord.values.T, color='green')
xycoord_blue= df_new_blue.loc[df_new_blue['well_name']=='G5', ['coord_x', 'coord_y']]#1500-1750  bluelight
xycoord_blue= xycoord_blue.iloc[::25, :]
axes.plot(*xycoord_blue.values.T, color='blue')
xycoord_post= df_new_post.loc[df_new_post['well_name']=='G5', ['coord_x', 'coord_y']]#4000post
xycoord_post= xycoord_post.iloc[::25, :]
axes.plot(*xycoord_post.values.T, color='red')
axes.set_xlim([450, 600])
axes.set_ylim([1650,1800])
plt.legend(title='conditions', loc='lower right', labels=['Pre-stimulus','blue light on ','Post-stimulus'])
#plot_stimuli(ax=ax, units='frames')
#plt.xlim(xmin = 0)
plt.xlabel('x_coordinates(μm)')
plt.ylabel('y_coordinates(μm')
plt.title(' Single-worm trajectory(first blue light exposure)')



df_new['conditions']=0
df_new_blue['conditions']=1
df_new_post['conditions']=2

df_concat=pd.concat([df_new, df_new_blue, df_new_post],keys=['pre-sti', 'bluelight','post-sti'], names=['name', 'Row ID'],ignore_index=True)


fig, ax = plt.subplots()
sns.lineplot(x='coord_x', y='coord_y',
                     hue='conditions', # stationary mode removed, forward mode=1, backward mode =-1
                     #style='worm_index', 
                     palette=["g", "b","r"],#'worm_strain',
                     data= df_concat,
                     #estimator=np.mean, ci='sd',
                     legend='full')
plt.legend(title='conditions', loc='lower right', labels=['Before stimulus','blue light','Post-stimulus'])
#plot_stimuli(ax=ax, units='frames')
#plt.xlim(xmin = 0)
plt.xlabel('x_coordinates(μm)')
plt.ylabel('y_coordinates(μm')

plt.title('Trajectory of a single worm  in well G5 in the first blue light stimulation')


feature_N = read_timeseries(file, only_wells=['G5'])
feature_N= filter_timeseries(feature_N)
feature_N_first_bluelight=feature_N.query('0< timestamp< 4000')
feature_N_first_bluelight['time(s)'] = feature_N_first_bluelight['timestamp'] / 25

from itertools import chain, repeat
new_col=list(chain(repeat(0, 1500), repeat(1, 250), repeat(2, 2249)))

feature_N_first_bluelight.insert(3, column='conditions', value=new_col)

fig, ax = plt.subplots()
ax=sns.lineplot(x='time(s)', y='speed',data= feature_N_first_bluelight, hue='conditions', # stationary mode removed, forward mode=1, backward mode =-1
                     #style='worm_index', 
                     palette=["g", "b","r"],
                     #estimator=np.mean, ci='sd',
                     legend='full')
plt.legend(title='conditions', loc='lower right', labels=['Pre-stimulus','blue light on ','Post-stimulus'])
ax.set_ylabel('speed (μm/s)')
plt.xlim(xmin = 0)
plot_stimuli(ax=ax, units='s')


feature_pre=feature_N_first_bluelight.query('0< timestamp< 1500')
feature_blue=feature_N_first_bluelight.query('1500< timestamp< 1750')
feature_post=feature_N_first_bluelight.query('1750< timestamp< 4000')




fig, ax = plt.subplots()
ax.plot(feature_pre['time(s)'],feature_pre['length'], color='green')
ax.plot(feature_blue['time(s)'],feature_blue['length'], color='blue')
ax.plot(feature_post['time(s)'],feature_post['length'], color='red')
plt.legend(title='conditions', loc='lower right', labels=['Pre-stimulus','blue light on ','Post-stimulus'])
ax.set_ylabel('curvature_midbody (rad/μm)')
plt.xlim(xmin = 0)
ax.set_xlabel('time (s)')
plot_stimuli(ax=ax, units='s')











