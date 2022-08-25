import os
import sys
import time
import warnings
from joblib import Parallel, delayed
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted, ns
from scipy import stats
from skimage import img_as_uint
from skimage.io import imsave, imread
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from matplotlib.colors import ListedColormap
import matplotlib as mpl

from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, split_mask, segment_object, process_u_csv, process_y_ave, combine_tlapse_data, interp_ty, fnames_to_time, calc_cmax
from cytomata.utils import setup_dirs, list_img_files, custom_styles, custom_palette


def process_fluo_images(img_dir, save_dir, segmt=False, segmt_factor=1, sb_microns=11, cmax='all'):
    """Analyze fluorescence images and generate figures."""
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    data = []
    if cmax == 'all':
        cmax = calc_cmax(img_dir)
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax == 'each':
            cmax_i = np.percentile(img, 99.999)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        plot_cell_img(den, None, fname, os.path.join(save_dir, 'denoised'), cmax=cmax_i, sb_microns=sb_microns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(save_dir, 'subtracted', fname + '.tiff'), img_as_uint(rescale_intensity(img)))
        rows = [{'i': fname, 'y': np.mean(img)}]
        if segmt:
            thr, regs, n = segment_object(img, segmt_local=True, factor=segmt_factor, rs=20)
            plot_cell_img(den, thr, fname, os.path.join(save_dir, 'roi'), cmax=cmax_i, sb_microns=sb_microns)
            rows = [{'i': fname, 'y': prop.mean_intensity, 'a': prop.area}
                    for prop in regionprops(regs, img)]
        data.extend(rows)
    df = pd.DataFrame(data)
    if segmt:
        df = df[(np.abs(stats.zscore(df['a'])) < 3)]
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


def combine_groups(root_dir, ch):
    df = pd.DataFrame(columns=['group', 'response'])
    for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0], alg=ns.IGNORECASE)):
        print(i, data_dir)
        y_csv = os.path.join(root_dir, data_dir, ch + '-results', 'y.csv')
        y_data = pd.read_csv(y_csv)
        rs = y_data['y'].values
        gr = np.full_like(rs, i+1)
        di = pd.DataFrame(np.column_stack([gr, rs]), columns=['group', 'response'])
        df = pd.concat([df, di], ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


def plot_groups(root_dir, group_labels, group_order, figsize=(16, 16)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.group.isin(group_order)]
    df['response'] = np.log10(df['response'])
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        mpl.colormaps.register(cmap=ListedColormap(custom_palette), name='custom')
        fig, ax = plt.subplots(figsize=figsize)
        sns.violinplot(x="group", y="response", data=df, order=group_order, hue_order=group_order, ax=ax,
                       linewidth=0, scale='count', inner=None, scale_hue=False, cut=0, zorder=0)
        # sns.stripplot(x="group", y="response", data=df, order=group_order, ax=ax,
        #                 jitter=0.1, size=2, linewidth=0, dodge=True, alpha=0.4, zorder=0)
        sns.pointplot(x="group", y="response", data=df, order=group_order, ax=ax,
                        estimator=np.mean, ci='sd', join=False, dodge=0.4, markers='.',
                        errwidth=2, capsize=0.05, scale=0.3, color='#212121', zorder=1)
        ax.set_xlabel('')
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Fluorescence (AU)')
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        ymin, ymax = ax.get_ylim()
        tick_range = np.arange(np.floor(ymin), 1)
        ax.yaxis.set_ticks(tick_range)
        ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        # ax.set_yscale('log')
        fig_name = 'y_' + '-'.join([str(int(g)) for g in df.group.unique()]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def combine_before_after(root_dir, ch):
    df = pd.DataFrame(columns=['group', 'timepoint', 'response'])
    for tpoint in [0, 1]:
        tp_dir = os.path.join(root_dir, str(tpoint))
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(tp_dir)][0])):
            y_csv = os.path.join(tp_dir, data_dir, ch + '-results', 'y.csv')
            y_data = pd.read_csv(y_csv)
            rs = y_data['y'].values
            gr = np.full_like(rs, i+1)
            tp = np.full_like(rs, tpoint)
            di = pd.DataFrame(np.column_stack([gr, tp, rs]), columns=['group', 'timepoint', 'response'])
            df = pd.concat([df, di], ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


def plot_before_after(root_dir, group_labels, group_order, figsize=(24, 8), tp_label=['t=0hr', 't=24hr']):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.group.isin(group_order)]
    fc_vals = {}
    for gr in df.group.unique():
        mean_before = df.loc[(df['group'] == gr) & (df['timepoint'] == 0), 'response'].mean()
        mean_after = df.loc[(df['group'] == gr) & (df['timepoint'] == 1), 'response'].mean()
        fc_vals[gr] = str(round(mean_after/mean_before, 2)) + r'$\times$'
    palette = ['#B0BEC5', '#1976D2']
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(x="group", y="response", hue='timepoint', data=df, order=group_order, ax=ax,
                      size=7, linewidth=0, dodge=True, alpha=0.8, zorder=0)
        sns.pointplot(x="group", y="response", hue='timepoint', data=df, order=group_order, ax=ax,
                      estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3,
                      capsize=0.1, scale=0.5, color='#212121', zorder=1)
        for i, gr in enumerate(group_order):
            y = df.loc[(df['group'] == gr), 'response'].max()
            m = df.response.max() * 0.05
            ax.plot([i-0.2, i-0.2, i+0.2, i+0.2], [y+m, y+m*1.5, y+m*1.5, y+m], lw=3, color='#212121')
            ax.text(i, y+m*2, fc_vals[gr], ha='center', va='bottom', color='#212121', fontsize=20)
        ax.set_xlabel('')
        ax.set_ylabel('Fluorescence (AU)')
        ax.set_xticklabels(group_labels)
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, tp_label, loc='best', prop={"size": 20}, frameon=True, shadow=False)
        fig_name = 'y_' + '-'.join([str(int(g)) for g in df.group.unique()]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def process_fluo_timelapse(img_dir, save_dir, u_csv=None, maskf=None, segmt=False, segmt_factor=1,
                           t_unit='s', ulabel='Light', sb_microns=11, cmax='all'):
    """Analyze fluorescence timelapse images and generate figures."""
    if cmax == 'all':
        cmax = calc_cmax(img_dir)
    u_df = None
    t_ann_img = []
    t = fnames_to_time(img_dir)
    if u_csv is not None and os.path.isfile(u_csv) and os.path.exists(u_csv):
        u_df, t_ann_img = process_u_csv(t, u_csv, save_dir)
    masks = [np.ones_like(imread(list_img_files(img_dir)[0]))]
    if maskf is not None:
        thr, masks = split_mask(maskf)
    y = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        fname = str(round(float(fname), 2))
        sig_ann = round(float(fname), 1) in t_ann_img
        img, raw, bkg, den = preprocess_img(imgf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup_dirs(os.path.join(save_dir, 'subtracted'))
            imsave(os.path.join(save_dir, 'subtracted', fname + '.tiff'), img_as_uint(rescale_intensity(img)))
        plot_bkg_profile(fname, save_dir, raw, bkg)
        plot_cell_img(den, None, fname, os.path.join(save_dir, 'denoised'), cmax=cmax, sb_microns=sb_microns)
        if i == 0:
            y0 = [np.mean(img[mask]) for mask in masks]
        if maskf is not None:
            plot_cell_img(den, thr, fname, os.path.join(save_dir, 'roi'), cmax=cmax, sb_microns=sb_microns)
        rows = [{'t': float(fname), 'y': np.mean(img[mask])/y0[n], 'n': n} for n, mask in enumerate(masks)]
        y.extend(rows)
    y_df = pd.DataFrame(y)
    y_df.to_csv(os.path.join(save_dir, 'y_raw.csv'), index=False)
    plot_uy(y_df, u_df, save_dir, t_unit=t_unit, ulabel=ulabel)
    process_y_ave(y_df, save_dir)


if __name__ == '__main__':
    # group_dir = '/media/phuong/Samsung USB/test/'
    # img_folder = 'imgs'
    # img_dir = os.path.join(group_dir, img_folder)
    # save_dir = os.path.join(group_dir, img_folder + '-results')
    # process_fluo_images(img_dir, save_dir, segmt=True, segmt_factor=1, sb_microns=110, cmax='all')

    ## Set Parameters ##
    root_dir = '/home/phuong/data/1-ifate/3-transcription/20220822/'
    img_folder = 'YFP'
    group_labels = [
        'Dark',
        '0.1 Hz BL',
        '0.5 Hz BL',
    ]
    group_order = [1, 2, 3]
    sb_microns = 110
    cmax = 'all'
    segmt = True
    segmt_factor = 1
    before_after = False

    if not before_after:
    ### Comparison of Groups ##
        for group_dir in natsorted(os.listdir(root_dir)):
            if group_dir == ".directory" or os.path.isfile(os.path.join(root_dir, group_dir)):
                continue
            print(group_dir)
            img_dir = os.path.join(root_dir, group_dir, img_folder)
            save_dir = os.path.join(root_dir, group_dir, img_folder + '-results')
            process_fluo_images(img_dir, save_dir, segmt=segmt, segmt_factor=1, sb_microns=sb_microns, cmax=cmax)
        combine_groups(root_dir, img_folder)
        plot_groups(root_dir, group_labels=group_labels, group_order=group_order, figsize=(len(group_labels)*8, 8))

    else:
    #### Before-After Fold Change ##
        for t in ['0', '1']:
            tp_dir = os.path.join(root_dir, t)
            for group_dir in natsorted(os.listdir(tp_dir)):
                if group_dir == ".directory":
                    continue
                print(group_dir)
                img_dir = os.path.join(tp_dir, group_dir, img_folder)
                save_dir = os.path.join(tp_dir, group_dir, img_folder + '-results')
                process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax)
        combine_before_after(root_dir, img_folder)
        plot_before_after(root_dir, group_labels, group_order=group_order, figsize=(len(group_labels)*8, 8))


    # root_dir = '/home/phuong/data/1-ifate/1-ddFP/ddFP/20220814_B3_RA_spike-t60_4/'
    # ch = 'mCherry'
    # img_dir = os.path.join(root_dir, ch)
    # u_csv = os.path.join(root_dir, 'u.csv')
    # u_csv = None
    # maskf = os.path.join(root_dir, 'Mask.tif')
    # # maskf = None
    # save_dir = os.path.join(root_dir, 'results')
    # process_fluo_timelapse(img_dir=img_dir, save_dir=save_dir, u_csv=u_csv, maskf=maskf, t_unit='s', ulabel='BL', sb_microns=11)

    # root_dir = '/home/phuong/data/ILID/ddFP/RA-16I/B3-sspBu_RA-16I_BL1-1s/mCherry-results/combined/'
    # # combine_tlapse_data(root_dir)
    # y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    # y_df['t'] = y_df['t'].round(decimals=0)
    # u_df = pd.read_csv(os.path.join(root_dir, 'u.csv'))
    # plot_uy(y_df, u_df['t'].values, u_df['u'].values, root_dir, t_unit='s', ulabel='BL')
