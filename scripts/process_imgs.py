import os
import sys
import json
import warnings
from collections import defaultdict
sys.path.append(os.path.abspath('../'))

import numpy as np
from tqdm import tqdm
from imageio import mimwrite
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.measure import regionprops

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted, ns
from scipy.interpolate import interp1d
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib.patches as patches
from skimage.io import imsave
from skimage.exposure import rescale_intensity
from skimage import img_as_uint

from cytomata.track import Sort, iou
from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, segment_object, segment_clusters, process_u_csv
from cytomata.utils import setup_dirs, list_img_files, custom_styles, custom_palette

import time


def count_cells(img_dir, save_dir, segmt_factor=1, remove_small=None, fill_holes=None):
    """Count number of cells in fluorescent image."""
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = np.percentile(img, 100)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr, reg, n = segment_object(den, factor=segmt_factor, rs=remove_small, fh=fill_holes)
        print(n)
        img_path = os.path.join(save_dir, fname + '.tif')
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     imsave(img_path, img_as_uint(img))
        img_save_dir = os.path.join(save_dir, 'den')
        cell_den = plot_cell_img(den, thr, fname, img_save_dir, cmax=cmax_i)


def process_US_timelapse(img_dir, save_dir, t_unit='s', sb_microns=None,
    cmax=None, segmt_factor=1, remove_small=None, fill_holes=None, adj_bright=False):
    """Analyze fluorescence timelapse images and generate figures."""
    # setup_dirs(os.path.join(save_dir, 'tracks'))
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    factor = segmt_factor
    tracker = Sort(max_age=3, min_hits=1)
    t = [float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in list_img_files(img_dir)]
    imgs = []
    data = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        ti = float(fname)
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.99)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        if adj_bright:
            a_reg = img[img > 0]
            if i == 0:
                kval = np.mean(a_reg[a_reg > np.percentile(a_reg, 95)])
            segmt_factor = factor * (kval/np.mean(a_reg[a_reg > np.percentile(a_reg, 95)]))
        thr, reg, n = segment_object(den, factor=segmt_factor, rs=remove_small, fh=fill_holes)
        ints = np.array([prop.mean_intensity for prop in regionprops(reg, img)])
        dets = np.array([prop.bbox for prop in regionprops(reg)])
        trks = tracker.update(dets)
        for trk in trks:  # Restructure data trajectory-wise
            id = int(trk[4])
            idx = np.argmax(np.array([iou(det, trk) for det in dets]))
            mi = float(ints[idx])
            bb = [float(det) for det in dets[idx]]
            data_row = {'id': id, 'time': ti, 'mean_int': mi, 'b_box': bb}
            data.append(data_row)
        img_path = os.path.join(save_dir, 'subtracted', fname + '.tiff')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(img_path, img_as_uint(rescale_intensity(img)))
        img_save_dir = os.path.join(save_dir, 'denoised')
        cell_den = plot_cell_img(den, None, fname, img_save_dir,
                cmax=cmax_i, t_unit=t_unit, sb_microns=sb_microns)
        img_save_dir = os.path.join(save_dir, 'outlined')
        cell_den = plot_cell_img(den, thr, fname, img_save_dir,
                cmax=cmax_i, t_unit=t_unit, sb_microns=sb_microns)
        imgs.append(cell_den)
    df = pd.DataFrame(data, index=t, columns=['id', 'time', 'mean_int', 'b_box'])
    df.dropna(how='all', inplace=True)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(save_dir, 'cell.gif'), imgs, fps=len(imgs)//10)


def plot_sc_tracks(root_dir, min_trk_len=150, figsize=(16, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    value_counts = df['id'].value_counts()
    keep = value_counts[value_counts >= min_trk_len].index.tolist()
    df = df[df['id'].isin(keep)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        # for idi in df['id'].unique():
        #     df_i = df.loc[(df['id'] == idi), ('time', 'mean_int')]
        sns.lineplot(data=df[['id', 'time', 'mean_int']], x='time', y='mean_int', hue='id', lw=2, palette=sns.color_palette("Blues", as_cmap=True))
        ax.set_xlabel('Time')
        ax.set_ylabel('AU')
        ax.get_legend().remove()
        fig_name = 'tracks.png'
        plt.savefig(os.path.join(save_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def process_fluo_timelapse(img_dir, save_dir, u_csv=None,
    t_unit='s', ulabel='BL', sb_microns=11, cmax=None,
    segmt=False, segmt_dots=False, segmt_mask=None, segmt_factor=1,
    remove_small=None, fill_holes=None, clear_border=None, adj_bright=False):
    """Analyze fluorescence timelapse images and generate figures."""
    if cmax is None:
        cmax = np.max([np.percentile(img_as_float(imread(imgf)), 99.9) for imgf in list_img_files(img_dir)])
    n_imgs = len(list_img_files(img_dir))
    t = [float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in list_img_files(img_dir)]
    y = []
    tu = []
    u = []
    t_ann_img = []
    imgs = []
    if os.path.isfile(u_csv) and os.path.exists(u_csv):
        tu, u, t_ann_img = process_u_csv(t, u_csv, save_dir)
    factor = segmt_factor
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        fname = str(round(float(fname), 2))
        img, raw, bkg, den = preprocess_img(imgf)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        yi = np.mean(img)
        if segmt:
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                seg_bound = img_as_float(imread(segmt_mask)) > 0
            if adj_bright:
                a_reg = img[img > 0]
                if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                    a_reg = seg_bound*img
                    a_reg = a_reg[a_reg > 0]
                if i == 0:
                    kval = np.mean(a_reg)
                segmt_factor = (np.mean(a_reg)/kval) + factor - 1
            if segmt_dots:
                thr = segment_clusters(den, factor=segmt_factor, rs=remove_small)
            else:
                thr = segment_object(den, factor=segmt_factor,
                    rs=remove_small, fh=fill_holes, cb=clear_border)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                thr *= seg_bound
            yi = np.mean(img[thr])
            if np.isnan(yi):
                yi = 0
        y.append(yi)
        sig_ann = round(float(fname), 1) in t_ann_img
        img_save_dir = os.path.join(save_dir, 'final')
        cell_img = plot_cell_img(den, thr, fname, img_save_dir,
            cmax, sig_ann, t_unit=t_unit, sb_microns=sb_microns)
        imgs.append(cell_img)
    plot_uy(t, y, tu, u, save_dir, t_unit=t_unit, ulabel=ulabel)
    data = np.column_stack((t, y))
    np.savetxt(os.path.join(save_dir, 'y.csv'),
        data, delimiter=',', header='t,y', comments='')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mimwrite(os.path.join(save_dir, 'cell.gif'), imgs, fps=len(imgs)//10)


def combine_uy(root_dir, fold_change=True, plot_u=True):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        if plot_u:
            fig, (ax0, ax) = plt.subplots(2, 1, sharex=True,
                figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        else:
            fig, ax = plt.subplots(figsize=(10,8)) 
        combined_t = pd.DataFrame()
        combined_y = pd.DataFrame()
        combined_tu = pd.DataFrame()
        combined_u = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0])):
            y_csv = os.path.join(root_dir, data_dir, 'y.csv')
            y_data = pd.read_csv(y_csv)
            t = y_data['t'].values
            y = y_data['y'].values
            yf = interp1d(t, y, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            t = pd.Series(t, index=t, name=i)
            combined_t = pd.concat([combined_t, t], axis=1)
            y = pd.Series([yf(ti) for ti in t], index=t, name=i)
            if fold_change:
                y = y/np.mean(y[:5])
            combined_y = pd.concat([combined_y, y], axis=1)
            ax.plot(y, color='#1976D2', alpha=1, linewidth=1)
            u_csv = os.path.join(root_dir, data_dir, 'u.csv')
            if plot_u:
                u_data = pd.read_csv(u_csv)
                tu = u_data['t'].values
                tu = pd.Series(tu, index=tu, name=i)
                u = pd.Series(u_data['u'].values, index=tu, name=i)
                combined_tu = pd.concat([combined_tu, tu], axis=1)
                combined_u = pd.concat([combined_u, u], axis=1)
        t_ave = combined_t.mean(axis=1).rename('t_ave')
        y_ave = combined_y.mean(axis=1).rename('y_ave')
        y_std = combined_y.std(axis=1).rename('y_std')
        y_sem = combined_y.sem(axis=1).rename('y_sem')
        if plot_u:
            tu_ave = combined_tu.mean(axis=1).rename('tu_ave')
            u_ave = combined_u.mean(axis=1).rename('u_ave')
            u_data = pd.concat([tu_ave, u_ave], axis=1).dropna()
            u_data.to_csv(os.path.join(root_dir, 'u.csv'), index=False)
        y_data = pd.concat([t_ave, y_ave, y_std, y_sem], axis=1).dropna()
        y_data.to_csv(os.path.join(root_dir, 'y.csv'), index=False)
        y_ave = y_data['y_ave']
        y_ci = y_data['y_sem']*1.96
        ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#648FFF', alpha=.2, label='95% CI')
        ax.plot(y_ave, color='#648FFF', label='Ave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        if fold_change:
            ax.set_ylabel('Fold Change')
        # ax.legend(loc='best')
        if plot_u:
            ax0.plot(tu, u, color='#648FFF')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        plot_name = 'y.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_fluo_images(img_dir, save_dir,
    sb_microns=11, cmax=None, segmt=False, segmt_dots=False,
    segmt_mask='', segmt_factor=1, remove_small=None,
    fill_holes=None, clear_border=None):
    """Analyze fluorescence 10x images and generate figures."""
    # if cmax is None:
    #     cmax_i = np.max([np.percentile(img_as_float(imread(imgf)), 99.999) for imgf in list_img_files(img_dir)])
    n_imgs = len(list_img_files(img_dir))
    y = []
    imgs = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        # fname = os.path.splitext(os.path.basename(imgf))[0]
        fname = str(i)
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.999)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        yi = np.mean(img)
        if segmt:
            segmt_mask = ''
            if segmt_mask is not None:
                segmt_mask = os.path.join(segmt_mask)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                seg_bound = img_as_float(imread(segmt_mask)) > 0
            if segmt_dots:
                thr = segment_clusters(den, factor=segmt_factor, rs=remove_small)
            else:
                thr = segment_object(den, factor=segmt_factor,
                    rs=remove_small, fh=fill_holes, cb=clear_border)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                thr *= seg_bound
            yi = np.mean(img[thr])
            if np.isnan(yi):
                yi = 0
        y.append(yi)
        img_save_dir = os.path.join(save_dir, 'final')
        cell_img = plot_cell_img(den, thr, fname, img_save_dir,
            cmax_i, sig_ann=False, t_unit=None, sb_microns=sb_microns)
    np.savetxt(os.path.join(save_dir, 'y.csv'),
        np.array(y), delimiter=',', header='y', comments='')


def combine_before_after(root_dir):
    df = pd.DataFrame(columns=['Group', 'Timepoint', 'Response'])
    for tpoint in [0, 24]:
        tp_dir = os.path.join(root_dir, 't' + str(tpoint))
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(tp_dir)][0])):
            y_csv = os.path.join(tp_dir, data_dir, 'results', 'y.csv')
            y_data = pd.read_csv(y_csv)
            rs = y_data['y'].values
            gr = np.full_like(rs, i+1)
            tp = np.full_like(rs, tpoint)
            print(data_dir, gr, rs)
            di = pd.DataFrame(np.column_stack([gr, tp, rs]), columns=['Group', 'Timepoint', 'Response'])
            df = df.append(di, ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


# def plot_before_after(root_dir, group_labels, exclude_groups=[]):
#     df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
#     df = df[~df.Group.isin(exclude_groups)]
#     df_fc = df.copy()
#     df_nm = df.copy()
#     ref0 = df.loc[(df['Group'] == 1) & (df['Timepoint'] == 0), 'Response'].mean()
#     ref24 = df.loc[(df['Group'] == 1) & (df['Timepoint'] == 24), 'Response'].mean()
#     for group in df.Group.unique():
#         ref0 = df.loc[(df['Group'] == 1) & (df['Timepoint'] == 0), 'Response'].mean()
#         ref24 = df.loc[(df['Group'] == 1) & (df['Timepoint'] == 24), 'Response'].mean()
#         df_nm.loc[(df_nm['Group'] == group) & (df_nm['Timepoint'] == 0), 'Response'] = df_nm.loc[(df_nm['Group'] == group) & (df_nm['Timepoint'] == 0), 'Response']/ref0
#         df_nm.loc[(df_nm['Group'] == group) & (df_nm['Timepoint'] == 24), 'Response'] = df_nm.loc[(df_nm['Group'] == group) & (df_nm['Timepoint'] == 24), 'Response']/ref24
#         ref = df.loc[(df['Group'] == group) & (df['Timepoint'] == 0), 'Response'].mean()
#         df_fc.loc[(df_fc['Group'] == group) & (df_fc['Timepoint'] == 24), 'Response'] = df_fc.loc[(df_fc['Group'] == group) & (df_fc['Timepoint'] == 24), 'Response']/ref
#     palette = ['#BBDEFB', '#1976D2']
#     with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
#         fig, (ax0, ax) = plt.subplots(2, 1, sharex=True, figsize=(24, 12), gridspec_kw={'height_ratios': [8, 8]})
#         df_fc = df_fc[df_fc.Timepoint == 24]
#         sns.stripplot(x="Group", y="Response", data=df_fc, ax=ax0, size=7, linewidth=0, dodge=True, alpha=0.8)
#         sns.pointplot(x="Group", y="Response", data=df_fc, ax=ax0, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
#         ax0.set_xlabel('')
#         ax0.set_ylabel('Log\nFold\nChange\n' + r'$\left [\frac{\bar{y}_{24 hr}}{\bar{y}_{0 hr}}\right ]$', rotation=0, ha="right")
#         ax0.set_yscale('log')
#         y_min = 10**(np.floor(np.log10(df_fc['Response'].min())))
#         y_max = 10**(np.ceil(np.log10(df_fc['Response'].max())))
#         ylim = (y_min, y_max)
#         ax0.set_ylim(ylim)
#         ax0.tick_params(which='minor', length=8, width=2)
#         ax0.tick_params(which='major', length=12, width=4)
#         df_nm = df_nm[df_nm.Timepoint == 24]
#         sns.stripplot(x="Group", y="Response", data=df_nm, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8)
#         sns.pointplot(x="Group", y="Response", data=df_nm, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
#         ax.axhline(y=1, color='#212121', linewidth=2, linestyle=(0, (5, 5)))
#         ax.axvline(x=0.5, color='#212121', linewidth=2, linestyle=(0, (5, 5)))
#         ax.set_xlabel('')
#         ax.set_xticklabels(group_labels)
#         ax.set_ylabel('Log\nFold\nDifference\n' + r'$\left [\frac{\bar{y}_{group@24hr}}{\bar{y}_{ctrl@24hr}}\right ]$', rotation=0, ha="right")
#         ax.set_yscale('log')
#         y_min = 10**(np.floor(np.log10(df_nm['Response'].min())))
#         y_max = 10**(np.ceil(np.log10(df_nm['Response'].max())))
#         ylim = (y_min, y_max)
#         ax.set_ylim(ylim)
#         ax.tick_params(which='minor', length=8, width=2)
#         ax.tick_params(which='major', length=12, width=4)
#         # handles, labels = ax.get_legend_handles_labels()
#         # ax.legend(handles, ['t=0hr', 't=24hr'], loc='best', prop={"size": 20})
#         fig_name = 'y_' + '-'.join([str(int(g)) for g in df.Group.unique()]) + '.png'
#         plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
#         plt.close()


# def plot_before_after(root_dir, group_labels, exclude_groups=[]):
#     df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
#     df = df[~df.Group.isin(exclude_groups)]
#     df_fc = df.copy()
#     for group in df.Group.unique():
#         ref = df.loc[(df['Group'] == group) & (df['Timepoint'] == 0), 'Response'].mean()
#         df_fc.loc[(df_fc['Group'] == group) & (df_fc['Timepoint'] == 24), 'Response'] = df_fc.loc[(df_fc['Group'] == group) & (df_fc['Timepoint'] == 24), 'Response']/ref
#     palette = ['#BBDEFB', '#1976D2']
#     with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
#         fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(24, 12), gridspec_kw={'height_ratios': [8, 8]})
#         sns.stripplot(x="Group", y="Response", hue='Timepoint', data=df, ax=ax0, size=7, linewidth=0, dodge=True, alpha=0.8)
#         sns.pointplot(x="Group", y="Response", hue='Timepoint', data=df, ax=ax0, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
#         ax0.set_xlabel('')
#         ax0.set_ylabel('AU', rotation=0, ha="right")
#         # ax0.set_yscale('log')
#         # y_min = 10**(np.floor(np.log10(df['Response'].min())))
#         # y_max = 10**(np.ceil(np.log10(df['Response'].max())))
#         # ylim = (y_min, y_max)
#         # ax0.set_ylim(ylim)
#         ax0.tick_params(which='minor', length=8, width=2)
#         ax0.tick_params(which='major', length=12, width=4)
#         handles, labels = ax0.get_legend_handles_labels()
#         ax0.legend(handles, ['t=0hr', 't=24hr'], loc='best', prop={"size": 20})
#         df_fc = df_fc[df_fc.Timepoint == 24]
#         sns.stripplot(x="Group", y="Response", data=df_fc, ax=ax1, size=7, linewidth=0, dodge=True, alpha=0.8)
#         sns.pointplot(x="Group", y="Response", data=df_fc, ax=ax1, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
#         ax1.set_xlabel('')
#         ax1.set_xticklabels(group_labels)
#         ax1.set_ylabel('Fold\nChange\n' + r'$\left [\frac{\bar{y}_{24 hr}}{\bar{y}_{0 hr}}\right ]$', rotation=0, ha="right")
#         # ax1.set_yscale('log')
#         # y_min = 10**(np.floor(np.log10(df_fc['Response'].min())))
#         # y_max = 10**(np.ceil(np.log10(df_fc['Response'].max())))
#         # ylim = (y_min, y_max)
#         # ax1.set_ylim(ylim)
#         ax1.tick_params(which='minor', length=8, width=2)
#         ax1.tick_params(which='major', length=12, width=4)
#         fig_name = 'y_' + '-'.join([str(int(g)) for g in df.Group.unique()]) + '.png'
#         plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
#         plt.close()


def plot_before_after(root_dir, group_labels, group_order, figsize=(24, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.Group.isin(group_order)]
    df_fc = df.copy()
    fc_vals = {}
    for gr in df.Group.unique():
        mean_t0 = df.loc[(df['Group'] == gr) & (df['Timepoint'] == 0), 'Response'].mean()
        mean_t24 = df.loc[(df['Group'] == gr) & (df['Timepoint'] == 24), 'Response'].mean()
        fc_vals[gr] = str(round(mean_t24/mean_t0, 2)) + r'$\times$'
    palette = ['#B0BEC5', '#1976D2']
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(x="Group", y="Response", hue='Timepoint', data=df, order=group_order, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8)
        sns.pointplot(x="Group", y="Response", hue='Timepoint', data=df, order=group_order, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
        for i, gr in enumerate(group_order):
            y = df.loc[(df['Group'] == gr), 'Response'].max()
            m = df.Response.max() * 0.05
            ax.plot([i-0.2, i-0.2, i+0.2, i+0.2], [y+m, y+m*1.5, y+m*1.5, y+m], lw=3, color='#212121')
            ax.text(i, y+m*2, fc_vals[gr], ha='center', va='bottom', color='#212121', fontsize=20)
        ax.set_xlabel('')
        ax.set_ylabel('AU')
        ax.set_xticklabels(group_labels)
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['t=0hr', 't=24hr'], loc='best', prop={"size": 20}, frameon=True, shadow=False)
        fig_name = 'y_' + '-'.join([str(int(g)) for g in df.Group.unique()]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def combine_groups(root_dir, ch):
    df = pd.DataFrame(columns=['Group', 'Response'])
    for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0], alg=ns.IGNORECASE)):
        print(data_dir)
        y_csv = os.path.join(root_dir, data_dir, ch + '-results', 'y.csv')
        y_data = pd.read_csv(y_csv)
        rs = y_data['y'].values
        gr = np.full_like(rs, i+1)
        di = pd.DataFrame(np.column_stack([gr, rs]), columns=['Group', 'Response'])
        df = df.append(di, ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


def plot_groups(root_dir, group_labels, group_order, figsize=(16, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.Group.isin(group_order)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(x="Group", y="Response", data=df, order=group_order, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8)
        sns.pointplot(x="Group", y="Response", data=df, order=group_order, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
        ax.set_xlabel('')
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('AU')
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        fig_name = 'y_' + '-'.join([str(int(g)) for g in df.Group.unique()]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()

def plot_lines(root_dir):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(20, 8))
        g = sns.lineplot(data=df, x="t", y="y", hue="h", estimator=np.mean, err_style="band", ci=68)
        g.legend_.set_title(None)
        ax.set_xlabel('Pulse Period')
        ax.set_ylabel('Fold Change in N/C Ratio')
        ax.set_xticks([2, 5, 10, 15, 30])
        # ax.tick_params(which='minor', length=8, width=2)
        # ax.tick_params(which='major', length=12, width=4)
        fig_name = 'y.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # root_dir = '/home/phuong/data/cell_count/'
    # img_dir = os.path.join(root_dir, 'imgs')
    # save_dir = os.path.join(root_dir, 'results')
    # count_cells(img_dir, save_dir, segmt_factor=3.5, remove_small=20, fill_holes=5)


    # root_dir = '/home/phuong/data/calcium/LOCCA/20210827_RGECO_LOCCA9_1/'
    # img_ch = 'mCherry'
    # for i in natsorted(os.listdir(os.path.join(root_dir, img_ch))):
    #     save_dir = os.path.join(root_dir, img_ch + '-results', str(i))
    #     img_dir = os.path.join(root_dir, img_ch, str(i))
    #     u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    #     mask = os.path.join(root_dir, 'mask.tif')
    #     process_fluo_timelapse(img_dir, save_dir, u_csv=u_csv,
    #         t_unit='s', ulabel='BL', sb_microns=110, cmax=None,
    #         segmt=False, segmt_dots=False, segmt_mask=mask, segmt_factor=3.0,
    #         remove_small=50, fill_holes=None, clear_border=None, adj_bright=True)

    # root_dir = '/home/phuong/data/calcium/LOCCA/20210827_combined/'
    # combine_uy(root_dir, fold_change=True, plot_u=True)
    
    root_dir = '/home/phuong/data/01_2MHz_200mV_20p_500ms/'
    channel = 'imgs'
    img_dir = os.path.join(root_dir, channel)
    save_dir = os.path.join(root_dir, channel + '-results')
    process_US_timelapse(img_dir, save_dir, t_unit=None, sb_microns=110,
        cmax=1.25, segmt_factor=3, remove_small=25, fill_holes=100, adj_bright=True)
    plot_sc_tracks(save_dir, min_trk_len=100, figsize=(16, 8))
    # process_fluo_images(img_dir, save_dir,
    #     sb_microns=None, cmax=None, segmt=True, segmt_dots=False, segmt_mask='',
    #     segmt_factor=4.0, remove_small=25, fill_holes=None, clear_border=None)

    # root_dir = '/home/phuong/data/GEX/20211210/'
    # ch = 'Default'
    # group_labels = [
    #     '10TAbs_mScI\nTAbd-VP64-NLS',
    #     '10TAbs_mScI\nTAbd-VP64-NES',
    # ]
    # combine_groups(root_dir, ch)
    # plot_groups(root_dir, group_labels, group_order=[2, 3], figsize=(len(group_labels)*6, 8))


    # for t in ['t0', 't24']:
    #     root_dir = '/home/phuong/data/GEX/20220121/{}/'.format(t)
    #     for group_dir in natsorted(os.listdir(root_dir)):
    #         if group_dir == ".directory":
    #             continue
    #         print(group_dir)
    #         img_dir = os.path.join(root_dir, group_dir, 'Default')
    #         save_dir = os.path.join(root_dir, group_dir, 'results')
    #         process_fluo_images(img_dir, save_dir,
    #             sb_microns=160, cmax=None, segmt=False, segmt_dots=False, segmt_mask=None,
    #             segmt_factor=1, remove_small=50, fill_holes=None, clear_border=None)


    # root_dir = '/home/phuong/data/GEX/20220121/'
    # combine_before_after(root_dir)
    # group_labels = [
    #     '6TetO\n+iLIDsl',
    #     '6TetO\n+iLIDsl\n+TetR-NLS',
    #     '6ZF3s-2TetO\n+iLIDsl',
    #     '6ZF3s-2TetO\n+iLIDsl\n+TetR-NLS'
    # ]
    # plot_before_after(root_dir, group_labels, group_order=[1, 2, 3, 4], figsize=(len(group_labels)*6, 8))


    # root_dir = '/home/phuong/data/'
    # plot_lines(root_dir)