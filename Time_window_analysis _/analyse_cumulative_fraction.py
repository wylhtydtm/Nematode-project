#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import time
import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tierpsytools.hydra.compile_metadata import (
    concatenate_days_metadata)

from helper import ( 
    find_motion_changes,
    read_metadata,
    plot_stimuli,
    load_bluelight_timeseries_from_results,
    just_load_one_timeseries,
    count_motion_modes,
    get_frac_motion_modes,
    HIRES_COLS，
    my_sum_bootstrap，get_frac_motion_modes_with_ci，plot_frac，plot_stacked_frac_mode
    )


# %% files and directories
# analysis dir
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


selected_meta_NoComp=select_meta(meta_updated, {'drug_type':'NoCompound'})

# %% read timeseries from results or disk

is_reload_timeseries_from_results = False

if is_reload_timeseries_from_results:
    # this uses tierpytools under the hood
    timeseries_df, hires_df = load_bluelight_timeseries_from_results(
        metadata_df,
        res_root_dir)
    # save to disk
    timeseries_df.to_hdf(timeseries_fname, 'timeseries_df', format='table')
    hires_df.to_hdf(timeseries_fname, 'hires_df', format='table')
else:  # from disk, then add columns
    # doing this now so that id I screw up I just have to load a fairly clean
    # dataframe from the saved file
    timeseries_df = pd.read_hdf(timeseries_fname, 'timeseries_df')
    hires_df = pd.read_hdf(timeseries_fname, 'hires_df')

# other feats to abs:
# a few features only make sense if we know ventral/dorsal
feats_to_abs = ['speed',
                'angular_velocity'
                'relative_to_body_speed_midbody',
                'd_speed',
                'relative_to_neck_angular_velocity_head_tip']

for feat in feats_to_abs:
    timeseries_df['abs_' + feat] = timeseries_df[feat].abs()

# %%% hand-picked features from the downsampled dataframe

plt.close('all')

feats_toplot = ['speed',
                'abs_speed',
                'angular_velocity',
                'abs_angular_velocity',
                'relative_to_body_speed_midbody',
                'abs_relative_to_body_speed_midbody',
                'abs_relative_to_neck_angular_velocity_head_tip',
                'speed_tail_base',
                'length',
                'major_axis',
                'd_speed',
                'head_tail_distance',
                'abs_angular_velocity_neck',
                'abs_angular_velocity_head_base',
                'abs_angular_velocity_hips',
                'abs_angular_velocity_tail_base',
                'abs_angular_velocity_midbody',
                'abs_angular_velocity_head_tip',
                'abs_angular_velocity_tail_tip']


with PdfPages(figures_dir / 'downsampled_feats.pdf', keep_empty=False) as pdf:
    for feat in tqdm(feats_toplot):
        fig, ax = plt.subplots()
        sns.lineplot(x='time_binned_s', y=feat,
                     hue='motion_mode',
                     style='worm_strain',
                     data=timeseries_df.query('motion_mode != 0'),
                     estimator=np.mean, ci='sd',
                     legend='full')
        plot_stimuli(ax=ax, units='s')
        pdf.savefig(fig)
        plt.close(fig)



# %%% hires feats plots

plt.close('all')
with PdfPages(figures_dir / 'hires_feats.pdf', keep_empty=False) as pdf:
    for col in tqdm(HIRES_COLS):
        if hires_df[col].dtype == 'float32':
            fig, ax = plt.subplots()
            sns.lineplot(x='timestamp', y=col,
                         hue='motion_mode',
                         style='worm_strain',
                         data=hires_df.query('motion_mode != 0'),
                         estimator='median', ci='sd',
                         legend='full', ax=ax)
            plot_stimuli(ax=ax, units='frames')
            pdf.savefig(fig)
            plt.close(fig)

with PdfPages(figures_dir / 'hires_feats_noerrs.pdf', keep_empty=False) as pdf:
    for col in tqdm(HIRES_COLS):
        if hires_df[col].dtype == 'float32':
            fig, ax = plt.subplots()
            sns.lineplot(x='timestamp', y=col,
                         hue='motion_mode',
                         style='worm_strain',
                         data=hires_df.query('motion_mode != 0'),
                         estimator='median', ci=None,
                         legend='full', ax=ax)
            plot_stimuli(ax=ax, units='frames')
            pdf.savefig(fig)
            plt.close(fig)


# %%% fraction motion modes

# get motion_mode stats
motion_mode_by_well = count_motion_modes(hires_df)

# plots with no error bars:

# aggregate data from all different wells, but keep strains separate
motion_mode = motion_mode_by_well.groupby(['worm_strain', 'timestamp'],
                                          observed=True).sum()
# compute fraction of worms in each motion mode (works with 'worm_strain' too)
frac_motion_mode = get_frac_motion_modes(motion_mode)

# %%% plots: stacked plot (AoE II style)
plt.close('all')
with PdfPages(figures_dir / 'motion_mode_frac_no_errs.pdf',
              keep_empty=False) as pdf:

    for strain_name, frmotmode_strain in frac_motion_mode.groupby(
            'worm_strain'):

        # look at area between curves here
        frmotmode_strain[['frac_worms_fw',
                          'frac_worms_st',
                          'frac_worms_bw',
                          'frac_worms_nan']].cumsum(axis=1).droplevel(
                              'worm_strain').plot()
        plt.gca().set_ylim([0, 1])
        plt.gca().set_ylabel('cumulative fraction')
        plt.gca().set_title(strain_name)
        plot_stimuli(units='frames', ax=plt.gca())
        pdf.savefig()
        plt.close()

        # just timeseries here instead
        frmotmode_strain.droplevel(
            'worm_strain').plot(
                y=['frac_worms_fw', 'frac_worms_st', 'frac_worms_bw'])
        plt.gca().set_ylim([0, 1])
        plt.gca().set_ylabel('fraction')
        plot_stimuli(units='frames', ax=plt.gca())
        plt.gca().set_title(strain_name)
        pdf.savefig()
        plt.close()



# %%%% example with seaborn:
tic = time.time()

foo = get_frac_motion_modes_with_ci(motion_mode_by_well, is_for_seaborn=True)

fig, ax = plt.subplots()
sns.lineplot(x='timestamp', y='frac_worms_fw', style='worm_strain',
             estimator='sum', ci=95,
             data=foo.reset_index(), ax=ax)
# frac_motion_mode.plot(y='frac_worms_fw', linestyle='--', ax=ax)  # check
sns.lineplot(x='timestamp', y='frac_worms_bw', style='worm_strain',
             estimator='sum', ci=95,
             data=foo.reset_index(), ax=ax)
# frac_motion_mode.plot(y='frac_worms_bw', linestyle='--', ax=ax)  # check
plot_stimuli(ax=ax, units='frames')
fig.savefig(figures_dir / 'frac_modes_sns.pdf')

print('Time elapsed: {}s'.format(time.time()-tic))

# %%%

tic = time.time()

is_recalculate_frac_motion_mode = False

if is_recalculate_frac_motion_mode:
    frac_motion_mode_with_ci = get_frac_motion_modes_with_ci(
        motion_mode_by_well)
    for col in ['frac_worms_bw_ci', 'frac_worms_st_ci',
                'frac_worms_fw_ci', 'frac_worms_nan_ci']:
        frac_motion_mode_with_ci[col+'_lower'] = \
            frac_motion_mode_with_ci[col].apply(lambda x: x[0])
        frac_motion_mode_with_ci[col+'_upper'] = \
            frac_motion_mode_with_ci[col].apply(lambda x: x[1])
        frac_motion_mode_with_ci.drop(columns=col, inplace=True)

    frac_motion_mode_with_ci.to_hdf(timeseries_fname,
                                    'frac_motion_mode_with_ci',
                                    format='table')
else:
    frac_motion_mode_with_ci = pd.read_hdf(timeseries_fname,
                                           'frac_motion_mode_with_ci')

fps = 25
frac_motion_mode_with_ci = frac_motion_mode_with_ci.reset_index()
frac_motion_mode_with_ci['time_s'] = (frac_motion_mode_with_ci['timestamp']
                                      / fps)
print('Time elapsed: {}s'.format(time.time()-tic))


# %%

fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
axs = [axs[-1], axs[0]]
plot_frac(frac_motion_mode_with_ci.reset_index(),
          ['frac_worms_fw', 'frac_worms_st', 'frac_worms_bw'], ax=axs)
for ax in axs:
    plot_stimuli(ax=ax, units='s')
plt.tight_layout()
fig.subplots_adjust()
fig.savefig(figures_dir / 'frac_worms_motion.pdf')


# %%

fig = plot_stacked_frac_mode(frac_motion_mode_with_ci)
with PdfPages(figures_dir / 'cumulative_frac_worms_motion.pdf') as pdf:
    for ff in fig:
        pdf.savefig(ff)



