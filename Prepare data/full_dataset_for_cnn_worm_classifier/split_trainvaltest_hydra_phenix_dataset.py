#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:13:23 2020

@author: lferiani

This script combines the manually annotated dataset of Hydra ROIs with
the older dataset that Avelino had used to train the tensorflow model in
Tierpsy 1.5.1. This older dataset probably combines phenix data with data
from other rigs as well...
"""

import tables
import numpy as np
import pandas as pd
from pathlib import Path


# %% where are things

root_dir = Path('/Users/liuziwei/Desktop/')
hd = root_dir / 'Hydra_Phenix_cnn'

# the hydra + phenix dataset
output_dataset_fname = hd / 'Hydra_Phenix_dataset.hdf5'
# %% parameters

test_size = 4000
val_size = 4000

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)

# %% randomly split dataset now

# shuffle an index

with tables.File(output_dataset_fname, 'r+') as fid:
    dataset_size = fid.get_node('/sample_data').shape[0]
ind = np.arange(dataset_size, dtype=int)
np.random.shuffle(ind)

# split it in 3 sets
split_ind = {'test': ind[:test_size],
             'val': ind[test_size:test_size+val_size],
             'train': ind[test_size+val_size:]}

# write them in groups
with tables.File(output_dataset_fname, 'r+') as fid:
    combined_annotations_df = pd.DataFrame(fid.get_node('/sample_data').read())
    combined_imgs = fid.get_node('/mask').read()
    for grp, grp_ind in split_ind.items():
        # create group
        if '/'+grp not in fid:c
            fid.create_group('/', grp)
        # save table. use index as it's same as img_row_ind
        ann_df = combined_annotations_df.loc[grp_ind].copy()
        ann_df.reset_index(drop=True, inplace=True)
        ann_df['img_row_id'] = np.arange(ann_df.shape[0]).astype(int)
        # save annotations table
        if '/'+grp+'/sample_data' in fid:
            fid.remove_node('/'+grp+'/sample_data', recursive=True)
        fid.create_table('/'+grp,
                         'sample_data',
                         obj=ann_df.to_records(index=False),
                         filters=FILTERS)
        # save rois
        if '/'+grp+'/mask' in fid:
            fid.remove_node('/'+grp+'/mask', recursive=True)
        fid.create_earray('/'+grp,
                          'mask',
                          obj=combined_imgs[grp_ind],
                          filters=FILTERS)
