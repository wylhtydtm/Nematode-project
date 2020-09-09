#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:30:39 2020

@author: liuziwei

Produce a scatter plot showing p values for all feeatures with different window combinations
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
import decimal
import scipy.stats 
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon
from scipy.stats import shapiro
from statsmodels.stats import multitest as smm
from statannot import add_stat_annotation
import math
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D 

#%%
hd= Path('/Volumes/Ashur Pro2/NoComp_rerun')
figures_dir = hd / 'Figure_scatter'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

NoComp_df=pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_10sTW/NoComp_5bluelightvideos_10TW_complied_filledmean.csv', index_col=False)
NoComp_df= NoComp_df.reset_index(drop=True)
NoComp_df.drop(columns='Unnamed: 0', inplace=True)

#Keep the wells that recorded across all windows
NoComp_df_1 = NoComp_df.groupby(by=['imgstore_name_bluelight', 'well_name']).filter(lambda x: len(x)==9)
NoComp_df_filtered=[x for window_id, x in NoComp_df_1.groupby(by='window_id')]
NoComp_df_updated= pd.concat(NoComp_df_filtered, axis=0).sort_index() 

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
                'major_axis_w_forward_90th',
                'd_major_axis_50th',
                'd_major_axis_w_forward_90th',
                'minor_axis_50th',
                'rel_to_body_speed_midbody_abs_90th',
                'd_rel_to_body_speed_midbody_abs_90th']

with PdfPages(hd / 'downsampled_feat_NoComp_10s_updated.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        NoComp_updated['window_id']= NoComp_updated['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=NoComp_updated)
        ax.set_xticks(np.arange(0, 8, step=1))
        add_stat_annotation(ax, data=NoComp_updated, x='window_id', y=feat, box_pairs=[('0.0','1.0'),('0.0', '2.0'),('1.0','2.0'),('1.0','7.0'),
                                                                                        ('3.0', '4.0'), ('4.0','5.0'),('3.0','5.0'),('6.0','7.0'), 
                                                                                        ('7.0','8.0'),('6.0','8.0')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        boxes = ax.artists
        NoComp_updated['window_id']= NoComp_updated['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = NoComp_updated['window_id'].tolist()[i]
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
hd= Path('/Users/liuziwei/Desktop/time_window')
figures_dir = hd / 'Figure_scatter'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

NoComp_df=pd.read_csv( hd / 'NoComp_TW_10s/NoComp_5bluelightvideos_10TW_complied_filledmean.csv', index_col=False)
NoComp_df= NoComp_df.reset_index(drop=True)

#Keep the wells that recorded across all windows
NoComp_df_1 = NoComp_df.groupby(by=['imgstore_name_bluelight', 'well_name']).filter(lambda x: len(x)==9)
NoComp_df_filtered=[x for window_id, x in NoComp_df_1.groupby(by='window_id')]
NoComp_df_complied= pd.concat(NoComp_df_filtered, axis=0).sort_index() 

NoComp_df_outpath = os.path.join(hd,'updated_NoComp_10s.csv')  
NoComp_df_complied.to_csv(NoComp_df_outpath)

NoComp_updated = pd.read_csv('/Users/liuziwei/Desktop/time_window/NoComp_TW_10s/updated_NoComp_10s.csv')

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

with PdfPages(hd / 'downsampled_feat_NoComp_10s_updated.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        NoComp_updated['window_id']= NoComp_updated['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=NoComp_updated)
        ax.set_xticks(np.arange(0, 8, step=1))
        add_stat_annotation(ax, data=NoComp_updated, x='window_id', y=feat, box_pairs=[('0.0','1.0'),('0.0', '2.0'),('1.0','2.0'),('1.0','7.0'),
                                                                                        ('3.0', '4.0'), ('4.0','5.0'),('3.0','5.0'),('6.0','7.0'), 
                                                                                        ('7.0','8.0'),('6.0','8.0')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        boxes = ax.artists
        NoComp_updated['window_id']= NoComp_updated['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = NoComp_updated['window_id'].tolist()[i]
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
NoComp_30df=pd.read_csv( hd / 'feature_summaries_30sTW/NoComp_5bluelightvideos_30sTW_complied_filledmean.csv', index_col=False)
NoComp_30df= NoComp_30df.reset_index(drop=True)
NoComp_30df.drop(columns='Unnamed: 0', inplace=True)

NoComp_30df_1 = NoComp_30df.groupby(by=['imgstore_name_bluelight', 'well_name']).filter(lambda x: len(x)==9)
NoComp_30df_filtered=[x for window_id, x in NoComp_30df_1.groupby(by='window_id')]
NoComp_30df_complied= pd.concat(NoComp_30df_filtered, axis=0).sort_index() 

NoComp_30df_outpath = os.path.join(hd,'feature_summaries_30sTW/updated_NoComp_30s.csv')  
NoComp_30df_complied.to_csv(NoComp_30df_outpath)


NoComp30_updated = pd.read_csv(hd / 'feature_summaries_30sTW/updated_NoComp_30s.csv')

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

with PdfPages(hd / 'downsampled_feat_NoComp_30s_updated.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        NoComp30_updated['window_id']= NoComp30_updated['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=NoComp30_updated)
        ax.set_xticks(np.arange(0, 8, step=1))
        add_stat_annotation(ax, data=NoComp30_updated, x='window_id', y=feat, box_pairs=[('0.0','1.0'),('0.0', '2.0'),('1.0','2.0'),('1.0','7.0'),
                                                                                        ('3.0', '4.0'), ('4.0','5.0'),('3.0','5.0'),('6.0','7.0'), 
                                                                                        ('7.0','8.0'),('6.0','8.0')],
                            test='t-test_paired', text_format='star', loc='inside', verbose=2)
        boxes = ax.artists
        NoComp30_updated['window_id']= NoComp30_updated['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = NoComp30_updated['window_id'].tolist()[i]
            if x== 1.0 or x== 4.0 or x ==7.0:
                box.set_facecolor('royalblue')
                box.set_edgecolor('black')
            else:
                box.set_facecolor('grey')
                box.set_edgecolor('black')
        pdf.savefig(fig)
        plt.close(fig)


#%%
NoComp_5ddf=pd.read_csv( '/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/NoComp_5bluelightvideos_10sTW_5slapse_complied_filledmean.csv', index_col=False)
NoComp_5ddf= NoComp_5ddf.reset_index(drop=True)
NoComp_5ddf.drop(columns='Unnamed: 0', inplace=True)

NoComp_5ddf_1 = NoComp_5ddf.groupby(by=['imgstore_name_bluelight', 'well_name']).filter(lambda x: len(x)==9)
NoComp_5ddf_filtered=[x for window_id, x in NoComp_5ddf_1.groupby(by='window_id')]
NoComp_5ddf_complied= pd.concat(NoComp_5ddf_filtered, axis=0).sort_index() 

NoComp_5ddf_outpath = os.path.join(hd,'updated_NoComp_shifted.csv')  
NoComp_5ddf_complied.to_csv(NoComp_5ddf_outpath)


NoComp5d_updated = pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/updated_NoComp_shifted.csv')
NoComp5d_updated.drop(columns='Unnamed: 0', inplace=True)


plt.close('all')

feats_toplot = ['length_90th',
                'speed_90th',
                'd_speed_90th',
                'curvature_head_abs_90th',
                'curvature_hips_abs_90th',
                'd_curvature_hips_abs_90th',
                'd_angular_velocity_head_base_abs_90th',
                'speed_neck_90th',
                'relative_to_body_angular_velocity_neck_abs_90th',
                'relative_to_body_angular_velocity_hips_abs_90th',              
                'motion_mode_paused_frequency',
                'curvature_neck_abs_90th',
                'curvature_hips_abs_90th',
                'angular_velocity_head_tip_abs_IQR',
                'angular_velocity_head_tip_abs_90th',
                'angular_velocity_abs_90th',
                'angular_velocity_head_base_abs_90th',
                'angular_velocity_head_base_abs_IQR',
                'curvature_midbody_abs_90th',
                'd_curvature_midbody_abs_90th',
                'd_curvature_neck_abs_90th']

with PdfPages(hd / 'downsampled_feat_NoComp_10s_5sdelay_updated.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        NoComp5d_updated['window_id']= NoComp5d_updated['window_id'].astype(str)
        order = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0','6.0','7.0','8.0']
        sns.boxplot(x='window_id', y=feat,data=NoComp5d_updated)
        ax.set_xticks(np.arange(0, 9, step=1))
        add_stat_annotation(ax, data=NoComp5d_updated, x='window_id', y=feat, box_pairs=[('0.0','1.0'),
                                                                                          ('3.0', '4.0'), ('6.0','7.0')],
                            test='t-test_paired', text_format='star', loc='inside', comparisons_correction=None, verbose=2)
        boxes = ax.artists
        NoComp5d_updated['window_id']= NoComp5d_updated['window_id'].astype(float)
        for i, box in enumerate(boxes):
            x = NoComp5d_updated['window_id'].tolist()[i]
            if x== 1.0 or x== 4.0 or x ==7.0:
                box.set_facecolor('royalblue')
                box.set_edgecolor('black')
            else:
                box.set_facecolor('grey')
                box.set_edgecolor('black')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)



#%%
 # double check the size of items in each window # 19 wells in NoComp_TW10s after filtering                           
df_cols = NoComp_df.columns[5:]                          
features_toplot=[feat for feat in df_cols if "path_curvature" not in feat]
n_feats= len(features_toplot) #2763 n total

p_value_threshold= 0.05

data_tw0 = NoComp_df_filtered[0].reset_index(drop=True)
data_tw1 = NoComp_df_filtered[1].reset_index(drop=True)
data_tw2 = NoComp_df_filtered[2].reset_index(drop=True)
data_tw3 = NoComp_df_filtered[3].reset_index(drop=True)
data_tw4 = NoComp_df_filtered[4].reset_index(drop=True)
data_tw5 = NoComp_df_filtered[5].reset_index(drop=True)
data_tw6 = NoComp_df_filtered[6].reset_index(drop=True)
data_tw7 = NoComp_df_filtered[7].reset_index(drop=True)
data_tw8 = NoComp_df_filtered[8].reset_index(drop=True)


#%%% Apply it for large dataset
def check_normality(data1, data2):
    normality_results = pd.DataFrame(data=None, columns=['stat','p_value'],  index = df_cols)
    for feat in df_cols:
        df=pd.DataFrame(columns=[feat])
        df[feat] = data1[feat] - data2[feat]
        stat, p_value = shapiro(df[feat])
        normality_results.loc[feat, 'stat'] = stat
        normality_results.loc[feat, 'p_value'] = p_value
    
    return normality_results

normality_tw01= check_normality(data_tw0, data_tw1)
sigdifffeats_tw01= normality_tw01[normality_tw01['p_value']> p_value_threshold]
sigdifffeats = sigdifffeats_tw01.index #1384 in total  to 420 features

sigdifffeats_outpath = os.path.join(hd,  'normality_test_tw01.csv')  
sigdifffeats_tw01.to_csv(sigdifffeats_outpath)


#%%%
tests = [ttest_ind, ttest_rel, mannwhitneyu, wilcoxon]   

def pairwise_test(test_data, control_data, tests= tests):
    
    dfcols= pd.DataFrame(data=None, columns = ['u_stats', 'p_values'])
    shared_colnames= list(dfcols.columns)
    
    for test in tests:
        p_decims = abs(int(decimal.Decimal(str(p_value_threshold)).as_tuple().exponent)) 
        test_name = str(test).split(' ')[1].split('.')[-1].split('(')[0].split('\'')[0]
        test_stats_df = pd.DataFrame(index=list(features_toplot), columns=shared_colnames)
    
        for feat in features_toplot:
            try:
                u_stats, p_values = test(test_data[feat], control_data[feat])
                test_stats_df.loc[feat, 'u_stats'] = u_stats
                test_stats_df.loc[feat, 'p_values'] = p_values
                #test_stats_df.reset_index(drop=True)
            except ValueError:
                pass  
        name_1 =[x for x in globals() if globals()[x] is test_data][0]
        name_2 =[x for x in globals() if globals()[x] is control_data][0]
        stats_outpath = os.path.join(hd, test_name, 'test_{}_{}vs{}'.format(test_name, name_1,name_2) +'_results.csv')  
        test_stats_df.to_csv(stats_outpath)
        
pairwise_test(data_tw0, data_tw1)
pairwise_test(data_tw0, data_tw2)
pairwise_test(data_tw3, data_tw4)
pairwise_test(data_tw3, data_tw5)
pairwise_test(data_tw6, data_tw7)
pairwise_test(data_tw6, data_tw8)
 # warnings.warn("Sample size too small for normal approximation.")

# Preprocess the ttest_rel, time_window,10
data_tw01= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_10sTW/t_test/ttest_rel/test_ttest_rel_data_tw0vsdata_tw1_results.csv',comment='#')
data_tw01= data_tw01.reset_index(drop=True)
#change unamed 0 column name to  features
data_tw01 = data_tw01.rename(columns={'Unnamed: 0': 'features'}) 

data_10_tw01= data_tw01.insert(1, 'tw_comparison','0_1') 
data_10s_tw01= data_tw01.drop(columns=['u_stats'])  
data_10stw_tw01 = data_10s_tw01.insert(2, 'time_window', '10')

del data_10_tw01 
del data_10stw_tw01
  
print(data_10s_tw01)

data_10s_tw01= data_10s_tw01.reset_index(drop=True)


#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05

test_pvalues_corrected_df = pd.DataFrame(index= list(data_10s_tw01['features']), columns=['updated_p','significant_ornot'])

p_values= data_10s_tw01.iloc[:,3]
_corrArray = smm.multipletests(p_values.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

#pvalues_corrected = _corrArray[1][_corrArray[0]] # Get p_values for features, 
test_pvalues_corrected_df['updated_p'] = _corrArray[1] # Add p_values to the dataframe
test_pvalues_corrected_df['significant_ornot'] = _corrArray[0]

#Calcalate the number of sgnificant features
pvalues_corrected = _corrArray[1][_corrArray[0]]
len(pvalues_corrected) #211 features   ( become139 values)

test_pvalues_corrected_df["-log10_pvalues"] = -np.log10(test_pvalues_corrected_df["updated_p"])


test_pvalues_corrected_df["Rank"]= test_pvalues_corrected_df["updated_p"].rank(ascending=True)
test_pvalues_corrected_df.sort_values("Rank", inplace = True) 

corrected_df_10s= test_pvalues_corrected_df[test_pvalues_corrected_df['updated_p']<= p_value_threshold]

corrected_outpath = os.path.join(hd,  't_test_paired_tw01_aftercorrection.csv')  
corrected_df_10s.to_csv(corrected_outpath)



#%%%
#Splitting features into different categories: Speed, velocity, postures, path, morphology,d_curvature, timeseries

features_names= list(pvalues02_corrected_df.index)

morphology_keywords = ['area','length','width','head'] #len=33
exclude_keywords_mor = ['d_', 'blob_','path']
morphology=[ft for ft in features_names if any([key in ft for key in morphology_keywords]) and all([key not in ft for key in exclude_keywords_mor])]

eignworms_keywords=['eigen_projection'] #224
eignworms=[ft for ft in features_names if any([key in ft for key in eignworms_keywords])]

speed_keywords =['speed'] #len=180
exclude_keywords_speed = ['d_speed','d_rel_to'] + morphology + eignworms
speed=[ft for ft in features_names if any([key in ft for key in speed_keywords]) and all([key not in ft for key in exclude_keywords_speed])]

blob_keywords=['blob','blob_box_length_','blob_area_IQR'] #len=432
exclude_keywords_blob = morphology + eignworms + speed
blob =[ft for ft in features_names if any([key in ft for key in blob_keywords]) and all([key not in ft for key in exclude_keywords_blob])] #448 len including blob area, size, comving, forward, backward

velocity_keywords =['velocity','ang_vel','radial_vel']  #len=800
exclude_keywords_velo = morphology + eignworms + speed +blob
velocity =[ft for ft in features_names if any([key in ft for key in velocity_keywords]) and all([key not in ft for key in exclude_keywords_velo])]

d_curvature_keywords = ['d_curvature_head','d_curvature_hips','d_curvature_midbody','d_curvature_neck','d_curvature_tail','d_curvature_mean','d_curvature_std']
d_curvature=[ft for ft in features_names if any([key in ft for key in d_curvature_keywords])]   #len=300                  

postures_keywords=['curvature', 'axis','quirkiness'] #len=412
exclude_keywords_post = d_curvature +morphology  + eignworms + speed +blob +velocity
postures= [ft for ft in features_names if any([key in ft for key in postures_keywords]) and all([key not in ft for key in exclude_keywords_post])]

path_keywords =['path','path_transit'] #40
path = [ft for ft in features_names if any([key in ft for key in path_keywords])]

d_keywords =['d_','d_speed'] #145
exclude_keywords_d = d_curvature + morphology + eignworms + speed +blob +velocity +postures +path
d_features= [ft for ft in features_names if any([key in ft for key in d_keywords]) and all([key not in ft for key in exclude_keywords_d])]

remaining = morphology + d_curvature + speed + velocity + postures + eignworms + blob +path + d_features

others = [x for x in features_names if any([x not in remaining])]

timeseries= others +d_features #343

#%%
#TW=30s, No Compound

data1_tw0 = NoComp_30df_filtered[0].reset_index(drop=True)
data1_tw1 = NoComp_30df_filtered[1].reset_index(drop=True)
data1_tw2 = NoComp_30df_filtered[2].reset_index(drop=True)
data1_tw3 = NoComp_30df_filtered[3].reset_index(drop=True)
data1_tw4 = NoComp_30df_filtered[4].reset_index(drop=True)
data1_tw5 = NoComp_30df_filtered[5].reset_index(drop=True)
data1_tw6 = NoComp_30df_filtered[6].reset_index(drop=True)
data1_tw7 = NoComp_30df_filtered[7].reset_index(drop=True)
data1_tw8 = NoComp_30df_filtered[8].reset_index(drop=True)


pairwise_test(data1_tw0, data1_tw1)
pairwise_test(data1_tw0, data1_tw2)
pairwise_test(data1_tw3, data1_tw4)
pairwise_test(data1_tw3, data1_tw5)
pairwise_test(data1_tw6, data1_tw7)
pairwise_test(data1_tw6, data1_tw8)

data1_tw01= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_30sTW/t_test/ttest_rel/test_ttest_rel_data1_tw0vsdata1_tw1_results.csv',comment='#')
data1_tw01= data1_tw01.reset_index(drop=True)
data1_tw01 = data1_tw01.rename(columns={'Unnamed: 0': 'features'}) 

data_30_tw01= data1_tw01.insert(1, 'tw_comparison','0_1') 
data_30s_tw01= data1_tw01.drop(columns=['u_stats'])  
data_30stw_tw01 = data_30s_tw01.insert(2, 'time_window', '30')

del data_30_tw01 
del data_30stw_tw01

print(data_30s_tw01)

data_30s_tw01["-log10_pvalues"] = -np.log10(data_30s_tw01["p_values"])

#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05

test_pvalues_corrected_df_30 = pd.DataFrame(index= list(data_30s_tw01['features']), columns=['updated_p','significant_ornot'])

p_values_30= data_30s_tw01.iloc[:,3]
_corrArray_30= smm.multipletests(p_values_30.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)
   #pvalues_corrected = _corrArray[1][_corrArray[0]] # Get p_values for features, 
test_pvalues_corrected_df_30['updated_p'] = _corrArray_30[1] # Add p_values to the dataframe
test_pvalues_corrected_df_30['significant_ornot'] = _corrArray_30[0]

#Calcalate the number of sgnificant features
pvalues_corrected_30 = _corrArray_30[1][_corrArray_30[0]]
len(pvalues_corrected_30) #433 features  #419 features

test_pvalues_corrected_df_30["-log10_pvalues"] = -np.log10(test_pvalues_corrected_df_30["updated_p"])


test_pvalues_corrected_df_30["Rank"]= test_pvalues_corrected_df_30["updated_p"].rank(ascending=True)
test_pvalues_corrected_df_30.sort_values("Rank", inplace = True) 

corrected_df_30s= test_pvalues_corrected_df_30[test_pvalues_corrected_df_30['updated_p']<= p_value_threshold]

corrected_outpath_30 = os.path.join(hd,  't_test_paired_tw01_30s aftercorrection.csv')  
corrected_df_30s.to_csv(corrected_outpath_30)
#%%

#5s shifted 10sTW
data2_tw0 = NoComp_5ddf_filtered[0].reset_index(drop=True)
data2_tw1 = NoComp_5ddf_filtered[1].reset_index(drop=True)
data2_tw2 = NoComp_5ddf_filtered[2].reset_index(drop=True)
data2_tw3 = NoComp_5ddf_filtered[3].reset_index(drop=True)
data2_tw4 = NoComp_5ddf_filtered[4].reset_index(drop=True)
data2_tw5 = NoComp_5ddf_filtered[5].reset_index(drop=True)
data2_tw6 = NoComp_5ddf_filtered[6].reset_index(drop=True)
data2_tw7 = NoComp_5ddf_filtered[7].reset_index(drop=True)
data2_tw8 = NoComp_5ddf_filtered[8].reset_index(drop=True)

pairwise_test(data2_tw0, data2_tw1)
pairwise_test(data2_tw0, data2_tw2)
pairwise_test(data2_tw3, data2_tw4)
pairwise_test(data2_tw3, data2_tw5)
pairwise_test(data2_tw6, data2_tw7)
pairwise_test(data2_tw6, data2_tw8)

data2_tw01= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/t_tests/ttest_rel/test_ttest_rel_data2_tw0vsdata2_tw1_results.csv',comment='#')
data2_tw01= data2_tw01.reset_index(drop=True)
data2_tw01 = data2_tw01.rename(columns={'Unnamed: 0': 'features'}) 

data_5s_tw01= data2_tw01.insert(1, 'tw_comparison','0_1') 
data_10s_5sshifted_tw01= data2_tw01.drop(columns=['u_stats'])  
data_10s_5s_tw01 = data_10s_5sshifted_tw01.insert(2, 'time_window', '10_5shifted')

del data_5s_tw01 
del data_10s_5s_tw01

print(data_10s_5sshifted_tw01)


#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05

test_pvalues_corrected_df_5s = pd.DataFrame(index= list(data_10s_5sshifted_tw01['features']), columns=['updated_p','significant_ornot'])

p_values_5s= data_10s_5sshifted_tw01.iloc[:,3]
_corrArray_5= smm.multipletests(p_values_5s.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

test_pvalues_corrected_df_5s['updated_p'] = _corrArray_5[1] # Add p_values to the dataframe
test_pvalues_corrected_df_5s['significant_ornot'] = _corrArray_5[0]

#Calcalate the number of sgnificant features
pvalues_corrected_5 = _corrArray_5[1][_corrArray_5[0]]
len(pvalues_corrected_5) #561 features   #479 features

test_pvalues_corrected_df_5s["-log10_pvalues"] = -np.log10(test_pvalues_corrected_df_5s["updated_p"])


test_pvalues_corrected_df_5s["Rank"]= test_pvalues_corrected_df_5s["updated_p"].rank(ascending=True)
test_pvalues_corrected_df_5s.sort_values("Rank", inplace = True) 

corrected_df_5s= test_pvalues_corrected_df_5s[test_pvalues_corrected_df_5s['updated_p']<= p_value_threshold]

corrected_outpath_5s = os.path.join(hd,  't_test_paired_tw01_shifted_10s_aftercorrection.csv')  
corrected_df_5s.to_csv(corrected_outpath_5s)
#%%
# Combine three windows pooled from 0.0-1.0 window comparision together in a dataframe for scatter plots

df_concat_1 =pd.concat([data_10s_tw01, data_30s_tw01], ignore_index= True)

p_threshold_conservative=-(math.log10(0.05))

data_30_tw01= data1_tw01.insert(1, 'tw_comparison','0_1') 
data_30s_tw01= data1_tw01.drop(columns=['u_stats'])  
data_30s_tw01= data_30s_tw01.drop([0])
data_30stw_tw01 = data_30s_tw01.insert(2, 'time_window', '30')

del data_30_tw01 
del data_30stw_tw01

print(data_30s_tw01)


# morphology_features + d_curvature_features + speed_features + velocity_features + postures_features 
#+ eignworms_features + blob_features +path_features  +timeseries_features 

sns.set(style = 'darkgrid')

ax= sns.scatterplot(test_pvalues_corrected_df.loc[morphology, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[d_curvature, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[speed, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[velocity, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[postures, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[eignworms, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[blob, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[path, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[timeseries, '-log10_pvalues'], test_pvalues_corrected_df_30.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=10s')
ax.set_ylabel('-log10_pvalues_for_time_window=30s')
ax.set_title('pre_stimulus_vs_bluelight')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')


#%%
#Plot for 10 and  5sshifted

sns.set(style = 'darkgrid')

ax= sns.scatterplot(test_pvalues_corrected_df.loc[morphology, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[d_curvature, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[speed, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[velocity, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[postures, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[eignworms, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[blob, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[path, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df.loc[timeseries, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=10s')
ax.set_ylabel('-log10_pvalues_for_time_window=10s_5sshifted')
ax.set_title('pre_stimulus_vs_bluelight')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')


#%%
#Plot for 30s and 5s shifted

sns.set(style = 'darkgrid')

ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[morphology, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[d_curvature, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[speed, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[velocity, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[postures, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[eignworms, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[blob, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[path, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(test_pvalues_corrected_df_30.loc[timeseries, '-log10_pvalues'], test_pvalues_corrected_df_5s.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=30s')
ax.set_ylabel('-log10_pvalues_for_time_window=10s_5sshifted')
ax.set_title('pre_stimulus_vs_bluelight')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')
#%%

# Preprocess the ttest_rel, time_window,10

# data_tw01= pd.read_csv('',comment='#')
# data_tw01= data_tw01.reset_index(drop=True)


# data_10_tw01= data_tw01.insert(1, 'tw_comparison','0_1') 
# data_10s_tw01= data_tw01.drop(columns=['u_stats'])  
# data_10s_tw01= data_10s_tw01.drop([0])
# data_10stw_tw01 = data_10s_tw01.insert(2, 'time_window', '10')

# del data_10_tw01 
# del data_10stw_tw01

# print(data_10s_tw01)
# data_10s_tw01= data_10s_tw01.reset_index(drop=True)


# #Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05

# test_pvalues_corrected_df = pd.DataFrame(index= list(data_10s_tw01['feat']), columns=['updated_p','significant_ornot'])

# p_values= data_10s_tw01.iloc[:,3]
# _corrArray = smm.multipletests(p_values.values, alpha=0.05, method='fdr_bh',\
#                                is_sorted=False, returnsorted=False)
#    #pvalues_corrected = _corrArray[1][_corrArray[0]] # Get p_values for features, 
# test_pvalues_corrected_df['updated_p'] = _corrArray[1] # Add p_values to the dataframe
# test_pvalues_corrected_df['significant_ornot'] = _corrArray[0]

# #Calcalate the number of sgnificant features
# pvalues_corrected = _corrArray[1][_corrArray[0]]
# len(pvalues_corrected) #211 features

# test_pvalues_corrected_df["-log10_pvalues"] = -np.log10(test_pvalues_corrected_df["updated_p"])

# legend1 = ax.legend(*scatter.legend_elements(num=1),
#                     loc="upper left", title="Ranking",frameon=False)
# ax.add_artist(legend1)

#%%
# ax = sns.scatterplot(x ='-log10_pvalues', y = '-log10_pvalues',
#                      hue = 'time_window' , data=df_concat_1)


# ax = sns.swarmplot(x ='-log10_pvalues', y = '-log10_pvalues',
#                      hue = 'time_window' , data=df_concat_1)


# plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
#             c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))

# plt.colorbar(ticks=range(6), label='digit value')
# plt.clim(-0.5, 5.5)

# fig,ax= plt.subplots()
# ax.scatter(x=df_concat_1.loc[df_concat_1['time_window']=='10','-log10_pvalues'],
#             y=df_concat_1.loc[df_concat_1['time_window']=='30','-log10_pvalues'],
#             alpha=0.5)



# ax.set_xlabel('-log10_pvalues_for_time_window=10s')
# ax.set_ylabel('-log10_pvalues_for_time_window=30s')

# line_1 = mlines.Line2D([0, 1], [0, 1], color='black')
# transform = ax.transAxes
# line_1.set_transform(transform)
# line_1.set_linestyle('--')
# ax.add_line(line_1)
# ax.axhline(y=4.74241088,color='r')
# ax.axvline(x=4.74241088,color='r')

# g = sns.replot(x ='tw_comparision', y = 'p_values',col = 'diff_tws',
#                hue = 'feat_types' , style='feat_types', kind='scatter', data=dfall_complied)


#%%%
# Preprocess the ttest_rel, time_window,0_2
data_tw02= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_10sTW/t_test/ttest_rel/test_ttest_rel_data_tw0vsdata_tw2_results.csv',comment='#')
data_tw02= data_tw02.reset_index(drop=True)
data_tw02 = data_tw02.rename(columns={'Unnamed: 0': 'features'}) 

data_10_tw02= data_tw02.insert(1, 'tw_comparison','0_2') 
data_10s_tw02= data_tw02.drop(columns=['u_stats'])  
data_10stw_tw02 = data_10s_tw02.insert(2, 'time_window', '10')

del data_10_tw02 
del data_10stw_tw02

print(data_10s_tw02)



drop = data_10s_tw02.loc[pd.isna(data_10s_tw02["p_values"]), :] #checking Nans
# Int64Index([312], dtype='int64') ,data_10s_tw02.loc[[312]], turn_inter_frequency, pvalue= NaN, removed
data_10s_tw02.drop(drop.index, inplace=True)  #2671 left

#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05
pvalues02_corrected_df = pd.DataFrame(index= list(data_10s_tw02['features']), columns=['updated_p','significant_ornot'])

p_values_tw02= data_10s_tw02.iloc[:,3]
_corrArray_tw02= smm.multipletests(p_values_tw02.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

pvalues02_corrected_df['updated_p'] = _corrArray_tw02[1] # Add p_values to the dataframe
pvalues02_corrected_df['significant_ornot'] = _corrArray_tw02[0]

#Calcalate the number of sgnificant features
pvalues02_corrected = _corrArray_tw02[1][_corrArray_tw02[0]]
len(pvalues02_corrected) #291 feats

pvalues02_corrected_df["-log10_pvalues"] = -np.log10(pvalues02_corrected_df["updated_p"])

#%%
#Preprocess the ttest_rel, window30, for comparing window0 and 2

data1_tw02= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_30sTW/t_test/ttest_rel/test_ttest_rel_data1_tw0vsdata1_tw2_results.csv',comment='#')
data1_tw02= data1_tw02.reset_index(drop=True)
data1_tw02 = data1_tw02.rename(columns={'Unnamed: 0': 'features'}) 

data_30_tw02= data1_tw02.insert(1, 'tw_comparison','0_2') 
data_30s_tw02= data1_tw02.drop(columns=['u_stats'])  
data_30stw_tw02 = data_30s_tw02.insert(2, 'time_window', '30')

del data_30_tw02 
del data_30stw_tw02

print(data_30s_tw02)

data_30s_tw02= data_30s_tw02.drop([311]) #drop 311  turn_inter_frequency becasue of missing pvalue somehow

data_30s_tw02.loc[pd.isna(data_30s_tw02["p_values"]), :]
data_30s_tw02.loc[[311]] #311  turn_inter_frequency pvalue= 1.0, removed
# Int64Index([312], dtype='int64') ,data_10s_tw02.loc[[312]], turn_inter_frequency, pvalue= NaN, removed


#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05
pvalues02_corrected_30df = pd.DataFrame(index= list(data_30s_tw02['features']), columns=['updated_p','significant_ornot'])

p_values_tw02_30s= data_30s_tw02.iloc[:,3]
_corrArray_tw02_30s= smm.multipletests(p_values_tw02_30s.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

pvalues02_corrected_30df['updated_p'] = _corrArray_tw02_30s[1] # Add p_values to the dataframe
pvalues02_corrected_30df['significant_ornot'] = _corrArray_tw02_30s[0]

#Calcalate the number of sgnificant features
pvalues02_corrected_30s= _corrArray_tw02_30s[1][_corrArray_tw02_30s[0]]
len(pvalues02_corrected_30s) #0 feature

pvalues02_corrected_30df["-log10_pvalues"] = -np.log10(pvalues02_corrected_30df["updated_p"])

#%%
# Preprocess the ttest_rel, time_window,10
data2_tw02= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/t_tests/ttest_rel/test_ttest_rel_data2_tw0vsdata2_tw2_results.csv',comment='#')
data2_tw02= data2_tw02.reset_index(drop=True)
data2_tw02 = data2_tw02.rename(columns={'Unnamed: 0': 'features'}) 

data_s5_tw02= data2_tw02.insert(1, 'tw_comparison','0_2') 
data_s5_tw02= data2_tw02.drop(columns=['u_stats'])  
data_s5tw_tw02 = data_s5_tw02.insert(2, 'time_window', '10_5sshifted')

del data_s5tw_tw02

print(data_s5_tw02)

drop_1=data_s5_tw02.loc[pd.isna(data_s5_tw02["p_values"]), :] #checking Nans
# Int64Index([312], dtype='int64') ,data_10s_tw02.loc[[312]], turn_inter_frequency, pvalue= NaN, removed
data_s5_tw02.drop(drop_1.index, inplace=True)  

#Perform Benjamini/Hochberg corrections for multiple comparisons to reduce false positive within each window. FDP less than 0.05
pvalues02_corrected_s5df = pd.DataFrame(index= list(data_s5_tw02['features']), columns=['updated_p','significant_ornot'])

p_values_tw02_s5= data_s5_tw02.iloc[:,3]
_corrArray_tw02_s5= smm.multipletests(p_values_tw02_s5.values, alpha=0.05, method='fdr_bh',\
                               is_sorted=False, returnsorted=False)

pvalues02_corrected_s5df['updated_p'] = _corrArray_tw02_s5[1] # Add p_values to the dataframe
pvalues02_corrected_s5df['significant_ornot'] = _corrArray_tw02_s5[0]

#Calcalate the number of sgnificant features
pvalues02_corrected_s5 = _corrArray_tw02_s5[1][_corrArray_tw02_s5[0]]
len(pvalues02_corrected_s5) #8 significant features/---0s


pvalues02_corrected_s5df.index[pvalues02_corrected_s5df['significant_ornot'] == True].tolist()
# 8 significant features are ['rel_to_body_ang_vel_tail_tip_w_paused_abs_90th','d_ang_vel_w_paused_abs_50th',
#'d_rel_to_body_ang_vel_tail_tip_w_paused_abs_90th', 'd_rel_to_body_ang_vel_tail_tip_w_paused_abs_IQR',
#'d_curvature_mean_tail_w_paused_abs_50th', 'd_width_head_base_w_paused_IQR','d_minor_axis_w_paused_90th','d_speed_head_base_w_paused_50th']


pvalues02_corrected_s5df["-log10_pvalues"] = -np.log10(pvalues02_corrected_s5df["updated_p"])




#%%

#Plotting scater plots

sns.set(style = 'darkgrid')

ax= sns.scatterplot(pvalues02_corrected_df.loc[morphology, '-log10_pvalues'], pvalues02_corrected_30df.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[d_curvature, '-log10_pvalues'], pvalues02_corrected_30df.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[speed, '-log10_pvalues'], pvalues02_corrected_30df.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[velocity, '-log10_pvalues'], pvalues02_corrected_30df.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[postures, '-log10_pvalues'], pvalues02_corrected_30df.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[eignworms, '-log10_pvalues'], pvalues02_corrected_30df.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[blob, '-log10_pvalues'], pvalues02_corrected_30df.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[path, '-log10_pvalues'], pvalues02_corrected_30df.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[timeseries, '-log10_pvalues'], pvalues02_corrected_30df.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=10s')
ax.set_ylabel('-log10_pvalues_for_time_window=30s')
ax.set_title('pre-stimulus_vs_post-stimulus')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')


#%%
sns.set(style = 'darkgrid')

ax= sns.scatterplot(pvalues02_corrected_df.loc[morphology, '-log10_pvalues'], pvalues02_corrected_s5df.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[d_curvature, '-log10_pvalues'], pvalues02_corrected_s5df.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[speed, '-log10_pvalues'], pvalues02_corrected_s5df.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[velocity, '-log10_pvalues'], pvalues02_corrected_s5df.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[postures, '-log10_pvalues'], pvalues02_corrected_s5df.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[eignworms, '-log10_pvalues'], pvalues02_corrected_s5df.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[blob, '-log10_pvalues'], pvalues02_corrected_s5df.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[path, '-log10_pvalues'], pvalues02_corrected_s5df.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_df.loc[timeseries, '-log10_pvalues'], pvalues02_corrected_s5df.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=10s')
ax.set_ylabel('-log10_pvalues_for_time_window=shifted_10s')
ax.set_title('pre-stimulus_vs_post-stimulus')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')

#%%
sns.set(style = 'darkgrid')

ax= sns.scatterplot(pvalues02_corrected_30df.loc[morphology, '-log10_pvalues'], pvalues02_corrected_s5df.loc[morphology, '-log10_pvalues'], label='morphology', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[d_curvature, '-log10_pvalues'], pvalues02_corrected_s5df.loc[d_curvature, '-log10_pvalues'], label='d_curvature', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[speed, '-log10_pvalues'], pvalues02_corrected_s5df.loc[speed, '-log10_pvalues'], label='speed', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[velocity, '-log10_pvalues'], pvalues02_corrected_s5df.loc[velocity, '-log10_pvalues'], label='velocity', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[postures, '-log10_pvalues'], pvalues02_corrected_s5df.loc[postures, '-log10_pvalues'], label='postures', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[eignworms, '-log10_pvalues'], pvalues02_corrected_s5df.loc[eignworms, '-log10_pvalues'], label='eignworms', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[blob, '-log10_pvalues'], pvalues02_corrected_s5df.loc[blob, '-log10_pvalues'], label='blob+d_blob', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[path, '-log10_pvalues'], pvalues02_corrected_s5df.loc[path, '-log10_pvalues'], label='path', alpha=0.8)
ax= sns.scatterplot(pvalues02_corrected_30df.loc[timeseries, '-log10_pvalues'], pvalues02_corrected_s5df.loc[timeseries, '-log10_pvalues'], label='time_series', alpha=0.8)

ax.set_xlabel('-log10_pvalues_for_time_window=30s')
ax.set_ylabel('-log10_pvalues_for_time_window=shifted_10s')
ax.set_title('pre-stimulus_vs_post-stimulus')
line_1 = Line2D([0, 1], [0, 1], color='black', alpha=0.5)
transform = ax.transAxes
line_1.set_transform(transform)
line_1.set_linestyle('--')
ax.add_line(line_1)
ax.axhline(y=1.3010299956639813,color='r',label='p_threshold')
ax.axvline(x=1.3010299956639813,color='r',label='p_threshold')
ax.legend(loc='center left', frameon=False,fontsize = 'x-small',bbox_to_anchor=(1, 0.55), title ='feature_sets')

