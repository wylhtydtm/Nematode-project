#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:53:13 2020

@author: liuziwei
"""
import pandas as pd
import numpy as np
import png
import os
from prepare_data import read_data_files


class IsMultiple():

    data_types = ['manual']  # from_tierpsy does not have multiple worms atm

    subset = 'validation'


    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.files = {}
        self.df = {data_type: {} for data_type in self.data_types}
        self.rois = {data_type: [] for data_type in self.data_types}
    

    def read_data(self):

        self.files = read_data_files(self.data_dir, self.subset, self.data_types)


    def check_multiple(self):

        for data_type in self.data_types:

            data = self.files[data_type]

            for i in range(len(data)):

                # get item
                file = data[i]

                # extract image and save
                roi = file['roi_full']
                if roi is None:
                    roi = file['roi_mask']
                self.rois[data_type].append(roi)

        
        # change into dataframe and check if multiple
        self.df[data_type] = pd.DataFrame(self.df[data_type], index=['n_skels']).T
        self.df[data_type]['is_multiple'] = self.df[data_type]['n_skels'] > 1
        self.df[data_type].index.name = 'worm_number'
        self.df[data_type] = self.df[data_type].reset_index()


    def write_to_file(self):

        for data_type, value in self.df.items():
            subdir = os.path.join(self.output_dir, data_type, self.subset)
            
            # to image
            img_dir = os.path.join(subdir, 'img')
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            for i in list(value.worm_number):
                filename = os.path.join(img_dir, '%s.png' % i)   
                roi = self.rois[data_type][i]
                png.from_array(roi, 'L').save(filename)
            
            # export CSV
            value.to_csv(os.path.join(subdir,'is_multiple.csv'), index=False)


    def run(self):
        self.read_data()
        self.check_multiple()
        self.write_to_file()


data_dir = '/Users/liuziwei/Downloads'  # change this to where the tar files are stored

output_dir = '/Users/liuziwei/Desktop'  # where you want to store your new files

IsMultiple(data_dir, output_dir).run()

