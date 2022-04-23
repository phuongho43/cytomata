import os
import sys
import time
import warnings
import itertools
from pathlib import Path
from joblib import Parallel, delayed
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from natsort import natsorted, ns
from skimage import img_as_uint
from skimage.io import imsave
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from matplotlib.colors import ListedColormap
import matplotlib as mpl

from cytomata.plot import plot_cell_img, plot_bkg_profile
from cytomata.process import preprocess_img, segment_object
from cytomata.utils import setup_dirs, list_img_files, custom_styles, custom_palette


def process_fluo_images(img_dir, save_dir, sb_microns=11, cmax=None):
    """Analyze fluorescence 10x images and generate figures."""
    def img_task(data, imgf):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.99)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        plot_cell_img(den, None, fname, os.path.join(save_dir, 'denoised'), cmax=cmax_i, sb_microns=sb_microns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(save_dir, 'subtracted', fname + '.tiff'), img_as_uint(rescale_intensity(img)))
        y = np.mean(img)
        data = {'i': fname, 'y': y}
        return data
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    data = []
    data = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(data, imgf) for imgf in list_img_files(img_dir))
    df = pd.DataFrame(data)
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


def plot_groups(root_dir, group_labels, group_order, figsize=(16, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.group.isin(group_order)]
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        mpl.colormaps.register(cmap=ListedColormap(custom_palette), name='custom')
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(x="group", y="response", data=df, order=group_order, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8, zorder=0)
        sns.pointplot(x="group", y="response", data=df, order=group_order, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121', zorder=1)
        ax.set_xlabel('')
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Fluorescence (AU)')
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
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
    df_fc = df.copy()
    fc_vals = {}
    for gr in df.group.unique():
        mean_before = df.loc[(df['group'] == gr) & (df['timepoint'] == 0), 'response'].mean()
        mean_after = df.loc[(df['group'] == gr) & (df['timepoint'] == 1), 'response'].mean()
        fc_vals[gr] = str(round(mean_after/mean_before, 2)) + r'$\times$'
    palette = ['#B0BEC5', '#1976D2']
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(x="group", y="response", hue='timepoint', data=df, order=group_order, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8, zorder=0)
        sns.pointplot(x="group", y="response", hue='timepoint', data=df, order=group_order, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121', zorder=1)
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


if __name__ == '__main__':
    ## Set Parameters ##
    root_dir = '/home/phuong/data/GEX/20220422/'
    img_folder = 'TxRed'
    group_labels = [
        'CaM + M13',
        'CaM-FUSN + M13-FUSN',
    ]
    group_order = [1, 2]
    sb_microns = 110
    cmax = None
    before_after = True
    

    if not before_after:
    #### Comparison of Groups ##
        for group_dir in natsorted(os.listdir(root_dir)):
            if group_dir == ".directory" or os.path.isfile(os.path.join(root_dir, group_dir)):
                continue
            print(group_dir)
            img_dir = os.path.join(root_dir, group_dir, img_folder)
            save_dir = os.path.join(root_dir, group_dir, img_folder + '-results')
            process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax)
        # combine_groups(root_dir, img_folder)
        # plot_groups(root_dir, group_labels=group_labels, group_order=group_order, figsize=(len(group_labels)*8, 8))

    else:
    #### Before-After Fold Change ##
        # for t in ['0', '1']:
        #     tp_dir = os.path.join(root_dir, t)
        #     for group_dir in natsorted(os.listdir(tp_dir)):
        #         if group_dir == ".directory":
        #             continue
        #         print(group_dir)
        #         img_dir = os.path.join(tp_dir, group_dir, img_folder)
        #         save_dir = os.path.join(tp_dir, group_dir, img_folder + '-results')
        #         process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax)
        combine_before_after(root_dir, img_folder)
        plot_before_after(root_dir, group_labels, group_order=group_order, figsize=(len(group_labels)*8, 8))