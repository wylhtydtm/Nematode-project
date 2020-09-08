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
from matplotlib import pyplot as plt


def plot_img_with_label(ann_df, imgs_fname, sample_size=20):
    d = (40, 120)
    with tables.File(imgs_fname, 'r') as fid:
        for _, row in ann_df.sample(n=20).iterrows():
            fig, ax = plt.subplots()
            img = fid.get_node('/mask')[row['img_row_id'],
                                        d[0]:d[1],
                                        d[0]:d[1]].astype('uint8')
            plt.imshow(img, cmap='gray')
            ax.set_title(f"worm? {row['is_worm']}")


def read_valid_annotations(fname):
    """
    Read valid subset of annotations, quash labels into is_worm
    """
    # read
    with pd.HDFStore(fname, 'r') as fid:
        df = fid['sample_data']
    valid_df = df.query('label_id > 0').copy()
    # quash
    valid_df['is_worm'] = valid_df['label_id'].map(is_worm_dict)
    return valid_df


def read_valid_rois(ann_df, imgs_fname):
    # use img_row_id as index into the full roi matrix
    with tables.File(imgs_fname, 'r') as fid:
        valid_imgs = fid.get_node(
            '/mask')[ann_df['img_row_id']].astype('uint8').copy()
    return valid_imgs


# %% where are things

root_dir = Path('/Users/lferiani/OneDrive - Imperial College London')
root_dir = root_dir / 'Analysis/Ziweis_NN/'
hd = root_dir / 'Hydra_Phenix_combined'

# the hydra dataset is split in two files, data + annotations
hydra_data_fname = root_dir / 'Hydra_dataset/worm_ROI_samples.hdf5'
hydra_annotations_fname = (
    root_dir / 'Hydra_dataset/hydra_ROI_annotationsonly.hdf5')

phenix_dataset_fname = (
    root_dir / 'Avelinos_manual_dataset/worm_ROI_samplesI.hdf5')

output_dataset_fname = hd / 'Hydra_Phenix_dataset.hdf5'

# %% parameters

# tweak size of ROI
roi_boundaries = (40, 120)  # make a 80px ROI

labels_dict = {1: 'not a worm',
               2: 'valid worm',
               3: 'difficult worm',
               4: 'worm aggregate',
               5: 'eggs',
               6: 'larvae',
               0: 'skipped'}
# which labels get pooled in definition of worm (True) vs not worm (False)
is_worm_dict = {1: False, 2: True, 3: True, 4: True, 5: False, 6: False}

test_size = 1000
val_size = 1000

FILTERS = tables.Filters(
    complevel=5, complib='zlib', shuffle=True, fletcher32=True)

plot_examples = True

# %% read  annotations, quash labels

hydra_valid_df = read_valid_annotations(hydra_annotations_fname)
hydra_valid_df['is_avelinos'] = False
phenix_valid_df = read_valid_annotations(phenix_dataset_fname)
phenix_valid_df['is_avelinos'] = True

# plot a few images with label
if plot_examples:
    plot_img_with_label(hydra_valid_df, hydra_data_fname)
    plot_img_with_label(phenix_valid_df, phenix_dataset_fname)

# %% get valid ROIs

# hydra first
hydra_valid_imgs = read_valid_rois(hydra_valid_df, hydra_data_fname)
# and now "reindex" the img_row_id
hydra_valid_df.rename(columns={'img_row_id': 'original_img_row_id'},
                      inplace=True)
# right now, the index is the one that matches the annotations to the imgs
hydra_valid_df.reset_index(drop=True, inplace=True)

# now phenix
phenix_valid_imgs = read_valid_rois(phenix_valid_df, phenix_dataset_fname)
# and now "reindex" the img_row_id
phenix_valid_df.rename(columns={'img_row_id': 'original_img_row_id'},
                       inplace=True)
# right now, the index is the one that matches the annotations to the imgs
phenix_valid_df.reset_index(drop=True, inplace=True)


# %% combine:

# concatenate annotations
combined_annotations_df = pd.concat((hydra_valid_df, phenix_valid_df),
                                    axis=0,
                                    sort=False,
                                    ignore_index=True)
combined_annotations_df['img_row_id'] = np.arange(
    combined_annotations_df.shape[0]).astype(int)

# concatenate ROIs
combined_imgs = np.concatenate((hydra_valid_imgs, phenix_valid_imgs), axis=0)

# write images
with tables.File(output_dataset_fname, 'w') as fid:
    # create table array
    # tab_dtypes = combined_annotations_df.dtype

    # save annotations table
    fid.create_table('/',
                     'sample_data',
                     obj=combined_annotations_df.to_records(index=False),
                     filters=FILTERS)
    # save rois
    fid.create_earray('/',
                      'mask',
                      obj=combined_imgs,
                      filters=FILTERS)
