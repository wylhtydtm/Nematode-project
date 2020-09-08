#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:34:10 2020

@author: liuziwei
"""

import os
import seaborn as sns
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statannot import add_stat_annotation


#%%
hd = Path('/Users/liuziwei/Desktop/time_window')

DMSO_complied=pd.read_csv('/Users/liuziwei/Desktop/SyngentaScreen/DMSO_5bluelightvideos_10sTW_complied.csv')

DMSO_complied = DMSO_complied.reset_index(drop=True)

DMSO_complied = [x for window_id, x in DMSO_complied.groupby(by='window_id')]

DMSO_complied = [x.fillna(x.mean()) for x in DMSO_complied]

DMSO_complied = pd.concat(DMSO_complied, axis=0).sort_index()


plt.close('all')

feats_toplot = ['speed_90th',
                'speed_10th',
                'ang_vel_midbody_abs_90th',
                'd_speed_90th',
                'd_speed_10th',
                'd_ang_vel_midbody_abs_90th',
                'speed_w_forward_90th',
                'speed_w_forward_10th',
                'speed_head_tip_w_forward_90th',
                'speed_head_base_w_forward_90th',
                'speed_tail_base_w_forward_90th',
                'speed_head_tip_w_backward_90th',
                'speed_head_base_w_backward_90th',
                'speed_tail_base_w_backward_90th',
                'quirkiness_50th',
                'width_midbody_50th',
                'motion_mode_paused_frequency',
                'motion_mode_paused_fraction',
                'motion_mode_forward_frequency',
                'motion_mode_forward_fraction',
                'motion_mode_backward_frequency',
                'motion_mode_backward_fraction',
                'curvature_mean_neck_abs_90th',
                'curvature_mean_midbody_abs_90th',
                'd_curvature_midbody_abs_90th',
                'd_curvature_neck_abs_90th',
                'd_curvature_neck_w_forward_abs_90th',
                'd_curvature_midbody_w_forward_abs_90th',
                'major_axis_IQR',
                'd_major_axis_50th',
                'd_major_axis_w_forward_90th',
                'minor_axis_50th',
                'rel_to_body_speed_midbody_abs_90th',
                'd_rel_to_body_speed_midbody_abs_90th']

with PdfPages(hd / 'downsampled_feat_DMSO_10TW.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        DMSO_complied_1['window_id']= DMSO_complied['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=DMSO_complied_1)
        ax.set_xticks(np.arange(0, 8, step=1))
        add_stat_annotation(ax, data=DMSO_complied, x='window_id', y=feat, box_pairs=[('0.0','1.0'),('3.0', '4.0'),
                                                                                        ('4.0', '5.0'), ('6.0','7.0'), ('7.0','8.0')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        boxes = ax.artists
        DMSO_complied_1['window_id']= DMSO_complied['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = DMSO_complied_1['window_id'].tolist()[i]
            if x== 1.0 or x== 4.0 or x ==7.0:
                box.set_facecolor('royalblue')
                box.set_edgecolor('black')
            else:
                box.set_facecolor('grey')
                box.set_edgecolor('black')
        #plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
        pdf.savefig(fig)
        plt.close(fig)
        
#%%


NoComp_30TW_complied_1  =pd.read_csv('/Users/liuziwei/Desktop/time_window/NoComp_5bluelightvideos_30sTW_complied_filledmean.csv')


import seaborn as sns
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statannot import add_stat_annotation

plt.close('all')

feats_toplot = ['speed_90th',
                'speed_10th',
                'ang_vel_midbody_abs_90th',
                'd_speed_90th',
                'd_speed_10th',
                'd_ang_vel_midbody_abs_90th',
                'speed_w_forward_90th',
                'speed_w_forward_10th',
                'speed_head_tip_w_forward_90th',
                'speed_head_base_w_forward_90th',
                'speed_tail_base_w_forward_90th',
                'speed_head_tip_w_backward_90th',
                'speed_head_base_w_backward_90th',
                'speed_tail_base_w_backward_90th',
                'quirkiness_50th',
                'width_midbody_50th',
                'motion_mode_paused_frequency',
                'motion_mode_paused_fraction',
                'motion_mode_forward_frequency',
                'motion_mode_forward_fraction',
                'motion_mode_backward_frequency',
                'motion_mode_backward_fraction',
                'curvature_mean_neck_abs_90th',
                'curvature_mean_midbody_abs_90th',
                'd_curvature_midbody_abs_90th',
                'd_curvature_neck_abs_90th',
                'd_curvature_neck_w_forward_abs_90th',
                'd_curvature_midbody_w_forward_abs_90th',
                'major_axis_IQR',
                'd_major_axis_50th',
                'd_major_axis_w_forward_90th',
                'minor_axis_50th',
                'rel_to_body_speed_midbody_abs_90th',
                'd_rel_to_body_speed_midbody_abs_90th']

with PdfPages(hd / 'downsampled_feat_NoComp_30TW.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        NoComp_30TW_complied_1['window_id']= NoComp_30TW_complied_1['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=NoComp_30TW_complied_1)
        ax.set_xticks(np.arange(0, 8, step=1))
        add_stat_annotation(ax, data=NoComp_30TW_complied_1 , x='window_id', y=feat, box_pairs=[('0.0','1.0'),('1.0','2.0'),
                                                                                        ('1.0', '7.0'), ('6.0','7.0'), ('7.0','8.0')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        boxes = ax.artists
        NoComp_30TW_complied_1['window_id']= NoComp_30TW_complied_1['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = NoComp_30TW_complied_1['window_id'].tolist()[i]
            if x== 1.0 or x== 4.0 or x ==7.0:
                box.set_facecolor('royalblue')
                box.set_edgecolor('black')
            else:
                box.set_facecolor('grey')
                box.set_edgecolor('black')
        #plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
        pdf.savefig(fig)
        plt.close(fig)

       

























