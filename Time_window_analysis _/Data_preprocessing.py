#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: liuziwei
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions
import os
#%% Input
root = Path('/Users/liuziwei/Imperial College London/Minga, Eleni - SyngentaScreen')
feat_file = root / 'SummaryFiles' / 'features_summaries_compiled.csv'
fname_file = root / 'SummaryFiles' / 'filenames_summaries_compiled.csv'

metadata_file =Path('/Users/liuziwei/Imperial College London/Minga, Eleni - SyngentaScreen/AuxiliaryFiles/new20200615metadata.csv')
moa_file = '/Users/em812/Data/Drugs/StrainScreens/AllCompoundsMoA.csv'

bad_well_cols = ['is_bad_well_from_gui', 'is_bad_well_misplaced_plate', 'is_bad_well_ledfailure']
#%% Read data
feat = pd.read_csv(feat_file, comment='#')
fname = pd.read_csv(fname_file, comment='#')

meta = pd.read_csv(metadata_file, index_col=None)
meta.loc[meta['drug_type'].isna(), 'drug_type'] = 'NoCompound'

# Match metadata to feature summaries
feat, meta = read_hydra_metadata(feat, fname, meta)
feat, meta = align_bluelight_conditions(feat, meta, how='outer')

del feat

meta_colnames = list(meta.columns)
print(meta_colnames)
#%% Choose the videos
# Keep only N2s
meta = meta[meta['worm_strain']=='N2']

# Remove wells missing bluelight conditions
imgstore_cols = [col for col in meta.columns if 'imgstore_name' in col]
miss = meta[imgstore_cols].isna().any(axis=1)
meta = meta.loc[~miss,:]

# Remove bad wells
meta = meta[meta['is_bad_well']== False]
#meta = meta.loc[~bad,:]

# Keep only DMSO or NoCompound controls
meta = meta[meta['drug_type'].isin(['DMSO','No','NoCompound'])]

# Keep only the plates that have all 4 N2 replicates both for DMSO and NoCompound
#Columns to keep:'imgstore_name_bluelight', 'imgstore_name_poststim', 'imgstore_name_prestim'
replic_meta = meta.groupby(by=['drug_type', 'imgstore_name_prestim'])

files_DMSO = pd.concat([group[imgstore_cols].iloc[[0]]
              for i,group in replic_meta
              if group.shape[0]==4 and i[0]=='DMSO'])
files_NoComp = pd.concat([group[imgstore_cols].iloc[[0]]
                for i,group in replic_meta
                if group.shape[0]==4 and i[0]=='NoCompound'])

# Sample a number of files per control type
n=5

files_DMSO = files_DMSO.sample(n)
files_NoComp = files_NoComp.sample(n)


#Save
hd = Path('/Users/liuziwei/Desktop/SyngentaScreen')
filesDMSO_metapath= os.path.join(hd, 'DMSO_file.csv')
files_DMSO.to_csv(filesDMSO_metapath, index=False)

filesNOComp_metapath= os.path.join(hd, 'NoComp_file.csv')
files_NoComp.to_csv(filesNOComp_metapath, index=False)

#%%

# data dir for one video
DMSO_root_dir_1 = '/Users/liuziwei/Desktop/SyngentaScreen/DMSO_5samples/syngenta_screen_run3_bluelight_20191212_155429.22956805/'

n_windows = 9
windows = list(range(n_windows))


summary_file_1 = [
    DMSO_root_dir_1+'features_summary_tierpsy_plate_20200518_175928_window_{}.csv'.format(i)
    for i in windows
        ]
 
import numpy as np   
summaries_1= []

for win in windows:
    print('Reading summary file for window {}...'.format(win))
    window = pd.read_csv(summary_file_1[win], comment='#')
    window = window[window['file_id']==0]
    window.insert(0, 'window_id', win*np.ones(window.shape[0])) # insert a column named window)id
    summaries_1.append(window)

del window

print('Reshaping summaries...')
summaries_1 = {
    well: df.reset_index(drop=True)
    for well,df in pd.concat(summaries_1, axis=0).groupby(by='well_name')
    }


df = meta[meta['imgstore_name_bluelight'].str.contains('20191212/syngenta_screen_run3_bluelight_20191212_155429.22956805')]
df = df[df['drug_type']=='DMSO']
keys_to_keep=list(df['well_name'])

summaries_filtered_1=[]

for key in keys_to_keep:
    print('filtering summary file'.format(key))
    summary_filtered_1 = summaries_1[key]
    summary_filtered_1.insert(1, 'imgstore_name_bluelight','20191212/syngenta_screen_run3_bluelight_20191212_155429.22956805')
    summaries_filtered_1.append(summary_filtered_1)

del summary_filtered_1

# now a column of window_id, a column of imgstore_name_blue_light, a column of well_number inserted in the dataframe.
#To look at individaul well

print('Reshaping summaries_filtered...')
summaries_1_filtered = {
    well: df.reset_index(drop=True)
    for well,df in pd.concat(summaries_filtered_1, axis=0).groupby(by='well_name')
    }



#concat them all together into a single dataframe

data=[]

nan_threshold = 0.2 # Threshold NaN proportion to drop feature from analysis  

for key in summaries_1_filtered:
    well = pd.DataFrame(summaries_1_filtered[key])
    well = well.dropna(axis=1, thresh=nan_threshold)
    data.append(well)  
    
df1_combined= pd.concat(data, axis=0,ignore_index=True)


#Save
df1_path= os.path.join(hd, 'syngenta_screen_run3_bluelight_20191212_155429.22956805_DMSO_10s.csv')

df1_combined.to_csv(df1_path, index=False)














