import os
import sys
import time
import warnings
from joblib import Parallel, delayed
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

from cytomata.track import Sort, iou
from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, segment_object, process_u_csv
from cytomata.utils import setup_dirs, list_img_files, custom_styles, custom_palette


def process_fluo_images(img_dir, save_dir, sb_microns=11, cmax=None,
    segmt=False, segmt_local=False, segmt_factor=1, remove_small=None):
    """Analyze fluorescence 10x images and generate figures."""
    def img_task(data, i, imgf):
        fname = str(i)
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.99)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        mi = np.mean(img)
        n = 0
        img_path = os.path.join(save_dir, 'subtracted', fname + '.tiff')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(img_path, img_as_uint(rescale_intensity(img)))
        img_save_dir = os.path.join(save_dir, 'denoised')
        cell_den = plot_cell_img(den, None, fname, img_save_dir, cmax=cmax_i, sb_microns=sb_microns)
        if segmt:
            thr, reg, n = segment_object(den, segmt_local=segmt_local, factor=segmt_factor, rs=remove_small)
            img_save_dir = os.path.join(save_dir, 'outlined')
            cell_den = plot_cell_img(den, thr, fname, img_save_dir, cmax=cmax_i, sb_microns=sb_microns)
            mi = np.mean(img[thr])
            if n < 10 or np.isnan(mi):
                mi = np.mean(img)
        data = {'fname': fname, 'mean_int': mi, 'num_cells': n}
        return data
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    ta = time.time()
    data = []
    # for i, imgf in enumerate(list_img_files(img_dir)):
    #     data = img_task(data, i, imgf)
    #     print(i)
    data = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(data, i, imgf) for i, imgf in enumerate(list_img_files(img_dir)))
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)
    print(time.time() - ta)


def combine_before_after(root_dir):
    df = pd.DataFrame(columns=['Group', 'Timepoint', 'Response'])
    for tpoint in [0, 24]:
        tp_dir = os.path.join(root_dir, 't' + str(tpoint))
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(tp_dir)][0])):
            y_csv = os.path.join(tp_dir, data_dir, 'results', 'y.csv')
            y_data = pd.read_csv(y_csv)
            rs = y_data['mean_int'].values
            gr = np.full_like(rs, i+1)
            tp = np.full_like(rs, tpoint)
            di = pd.DataFrame(np.column_stack([gr, tp, rs]), columns=['Group', 'Timepoint', 'Response'])
            df = pd.concat([df, di], ignore_index=True)
            # df = df.append(di, ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


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
        print(i, data_dir)
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
    root_dir = '/home/phuong/data/GEX/20220403_TetO-YB-NLS-mScI_TetR-VP64-mTq2-NES/'
    img_folder = 'TxRed'
    img_dir = os.path.join(root_dir, img_folder)
    save_dir = os.path.join(root_dir, img_folder + '-results')
    process_fluo_images(img_dir, save_dir, sb_microns=110, cmax=None,
        segmt=False, segmt_local=False, segmt_factor=1, remove_small=50)
    # root_dir = '/home/phuong/data/GEX/20220125/'
    # img_folder = 'TxRed'
    # group_labels = [
    #     'LOV2',
    #     'LOV2 + Lck',
    #     'LOV2 + cAID'
    # ]
    # combine_groups(root_dir, img_folder)
    # plot_groups(root_dir, group_labels, group_order=[1, 2, 3], figsize=(len(group_labels)*8, 8))


    # for t in ['t0', 't24']:
    #     root_dir = '/home/phuong/data/GEX/20220311/{}/'.format(t)
    #     for group_dir in natsorted(os.listdir(root_dir)):
    #         if group_dir == ".directory":
    #             continue
    #         print(group_dir)
    #         img_dir = os.path.join(root_dir, group_dir, 'TxRed')
    #         save_dir = os.path.join(root_dir, group_dir, 'results')
    #         process_fluo_images(img_dir, save_dir, sb_microns=110, cmax_all=False,
    #             segmt=False, segmt_local=False, segmt_factor=1, remove_small=50)
    # root_dir = '/home/phuong/data/GEX/20220311/'
    # combine_before_after(root_dir)
    # group_labels = [
    #     'BL 1s per 25s',
    #     'BL 1s per 20s',
    #     'BL 1s per 15s',
    #     'BL 1s per 10s',
    # ]
    # plot_before_after(root_dir, group_labels, group_order=[4, 3, 2, 1], figsize=(len(group_labels)*6, 8))


    # root_dir = '/home/phuong/data/'
    # plot_lines(root_dir)


    # root_dir = '/home/phuong/data/cell_count/'
    # img_dir = os.path.join(root_dir, 'imgs')
    # save_dir = os.path.join(root_dir, 'results')
    # count_cells(img_dir, save_dir, segmt_factor=3.5, remove_small=20, fill_holes=5)