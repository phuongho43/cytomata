import os
import sys
import warnings
sys.path.append(os.path.abspath('../'))

import numpy as np
from tqdm import tqdm
from imageio import mimwrite
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from scipy.interpolate import interp1d

from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_uy
from cytomata.process import preprocess_img, segment_object, segment_clusters, process_u_csv
from cytomata.utils import list_img_files, custom_styles, custom_palette

def iter_cb(img, prog):
    return False

def process_fluo_timelapse(img_dir, save_dir, u_csv=None,
    t_unit='s', ulabel='BL', sb_microns=11, cmax=None,
    segmt=False, segmt_dots=False, segmt_mask=None, segmt_factor=1,
    remove_small=None, fill_holes=None, clear_border=None, adj_bright=False, iter_cb=iter_cb):
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
        cell_img = plot_cell_img(den, thr, fname, save_dir,
            cmax, sig_ann, t_unit=t_unit, sb_microns=sb_microns)
        imgs.append(cell_img)
        prog = (i+1)/n_imgs * 100
        if iter_cb(cell_img, prog):
            break
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
        ax.fill_between(t, (y_ave - y_ci), (y_ave + y_ci), color='#1976D2', alpha=.2, label='95% CI')
        ax.plot(y_ave, color='#1976D2', label='Ave')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('AU')
        if fold_change:
            ax.set_ylabel('Fold Change')
        # ax.legend(loc='best')
        if plot_u:
            ax0.plot(tu, u, color='#1976D2')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel('BL')
        plot_name = 'y.png'
        fig.savefig(os.path.join(root_dir, plot_name),
            dpi=100, bbox_inches='tight', transparent=False)
        plt.close(fig)


def process_fluo_images(img_dir, save_dir,
    sb_microns=11, cmax=None, segmt=False, segmt_dots=False,
    segmt_mask_dir='', segmt_factor=1, remove_small=None,
    fill_holes=None, clear_border=None, quant='sum', iter_cb=iter_cb):
    """Analyze fluorescence 10x images and generate figures."""
    if cmax is None:
        cmax = np.max([np.percentile(img_as_float(imread(imgf)), 99.999) for imgf in list_img_files(img_dir)])
    n_imgs = len(list_img_files(img_dir))
    y = []
    imgs = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        # fname = os.path.splitext(os.path.basename(imgf))[0]
        fname = str(i)
        img, raw, bkg, den = preprocess_img(imgf)
        cmax = np.percentile(img, 99.999)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        thr = None
        if quant == 'sum':
            yi = np.sum(img)
        else:
            yi = np.mean(img)
        if segmt:
            segmt_mask = ''
            if segmt_mask_dir is not None:
                segmt_mask = os.path.join(segmt_mask_dir, fname)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                seg_bound = img_as_float(imread(segmt_mask)) > 0
            if segmt_dots:
                thr = segment_clusters(den, factor=segmt_factor, rs=remove_small)
            else:
                thr = segment_object(den, factor=segmt_factor,
                    rs=remove_small, fh=fill_holes, cb=clear_border)
            if os.path.isfile(segmt_mask) and os.path.exists(segmt_mask):
                thr *= seg_bound
            if quant == 'sum':
                yi = np.sum(img[thr])
            else:
                yi = np.mean(img[thr])
            if np.isnan(yi):
                yi = 0
        y.append(yi)
        cell_img = plot_cell_img(den, thr, fname, save_dir,
            cmax, sig_ann=False, t_unit=None, sb_microns=sb_microns)
        prog = (i+1)/n_imgs * 100
        if iter_cb(cell_img, prog):
            break
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
            di = pd.DataFrame(np.column_stack([gr, tp, rs]), columns=['Group', 'Timepoint', 'Response'])
            df = df.append(di, ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)



def compare_before_after(root_dir):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    groups = [1, 2, 3, 4, 5, 6]
    # groups = range(1, 13)
    df = df.loc[df['Group'].isin(groups)]
    t0_colnames = [0]
    t24_colnames = [24]
    for group in groups:
        for t0, t24 in zip(t0_colnames, t24_colnames):
            ref = df.loc[(df['Group'] == group) & (df['Timepoint'] == t0), 'Response'].mean()
            df.loc[(df['Group'] == group) & (df['Timepoint'] == t0), 'Response'] = df.loc[(df['Group'] == group) & (df['Timepoint'] == t0), 'Response']/ref
            df.loc[(df['Group'] == group) & (df['Timepoint'] == t24), 'Response'] = df.loc[(df['Group'] == group) & (df['Timepoint'] == t24), 'Response']/ref
    palette = ['#BBDEFB', '#1976D2']
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        fig, ax = plt.subplots(figsize=(24,8))
        # ax = sns.boxplot(x="Group", y="Response", hue="Timepoint", data=df, whis=np.inf, linewidth=3, fliersize=0)
        sns.stripplot(x="Group", y="Response", hue="Timepoint", data=df, ax=ax, size=7, linewidth=0, dodge=True, alpha=0.8)
        sns.pointplot(x="Group", y="Response", hue="Timepoint", data=df, ax=ax, estimator=np.mean, ci=99, join=False, dodge=0.4, markers='.', errwidth=3, capsize=0.1, scale=0.5, color='#212121')
        ax.set_xticklabels([
            '8ZF3s-mGd\nZF3-iLIDslow\nsspBn-NLS-p65',
            '8ZF3s-mGd\nZF3-NES-iLIDslow\nsspBn-NLS-p65',
            '8ZF3s-mGd\nZF3-iLIDslow\nsspBn-NLS-p65\nZF3-NLS',
            '8ZF3s-mGd\nZF3-iLIDslow\nsspBn-NLS-p65\nZF3-LANS',
            '8ZF3s-mGd\nZF3-NLS-Zdk\np65-NLS-LOV2',
            '8ZF3s-mGd\nZF3-NLS-Zdk\np65-NLS-LOV2\n(dark)'
        ])
        # ax.set_xticklabels([
        #     '6TetO-mScI',
        #     '6TetO-mScI\nTetR-VPR',
        # ])
        ax.set_xlabel('')
        ax.set_ylabel('Fold Change')
        # ax.set_yscale('log')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['t=0hr', 't=24hr'], loc='best', prop={"size": 20})
        fig_name = 'y_fc_' + '-'.join([str(g) for g in groups]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


def barplot_expts(root_dir):
    y_data = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.boxplot(x="System", y="Response", data=y_data, whis=np.inf, linewidth=3, fliersize=0)
        g = sns.stripplot(x="System", y="Response", data=y_data, ax=ax, size=10, linewidth=1, dodge=True)
        # ax.set_yscale("log")
        # g.ax.set_xticks([-0.2, 1.2])
        # plt.legend(loc='upper center', prop={"size": 20})
        ax.set_xticklabels(["ZF3-p65", "ZF3-p65\n+ZF3"])
        ax.set_ylabel('Ave Fl.')
        ax.set_xlabel('')
        plt.savefig(os.path.join(root_dir, 'y.png'), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # root_dir = '/home/phuong/data/calcium/CaM-M13/GEX/20210526_20210526_6TetO-mScl_M13-NLS-VPR_TetR-NLS-CaM/'
    # img_ch = 'TxRed'
    # for i in natsorted(os.listdir(os.path.join(root_dir, img_ch))):
    #     save_dir = os.path.join(root_dir, img_ch + '-results', str(i))
    #     img_dir = os.path.join(root_dir, img_ch, str(i))
    #     u_csv = os.path.join(root_dir, 'u{}.csv'.format(i))
    #     mask = os.path.join(root_dir, 'mask.tif')
    #     process_fluo_timelapse(img_dir, save_dir, u_csv='',
    #         t_unit='s', ulabel='BL', sb_microns=110, cmax=None,
    #         segmt=False, segmt_dots=False, segmt_mask=mask, segmt_factor=2.0,
    #         remove_small=50, fill_holes=None, clear_border=None, adj_bright=True)

    # root_dir = '/home/phuong/data/ILID/ddFP/RA-16I/B3-sspBu_RA-16I_BL1-1s/mCherry-results/'
    # combine_uy(root_dir, fold_change=True, plot_u=True)
    
    root_dir = '/home/phuong/data/FPs/NLS/20210806_mycA6nls-mTq2/'
    img_dir = os.path.join(root_dir, 'Default')
    save_dir = os.path.join(root_dir, 'results')
    process_fluo_images(img_dir, save_dir,
        sb_microns=16, cmax=None, segmt=False, segmt_dots=False,
        segmt_mask_dir='', segmt_factor=1.0, remove_small=50, fill_holes=None,
        clear_border=None, quant='mean')

    # root_dir = '/home/phuong/data/calcium/CaM-M13/GEX/20210526_CaM-M13/t0/'
    # for group_dir in natsorted(os.listdir(root_dir)):
    #     if group_dir == ".directory":
    #         continue
    #     print(group_dir)
    #     img_dir = os.path.join(root_dir, group_dir, 'Default')
    #     save_dir = os.path.join(root_dir, group_dir, 'results')
    #     process_fluo_images(img_dir, save_dir,
    #         sb_microns=110, cmax=None, segmt=False, segmt_dots=False,
    #         segmt_mask_dir='', segmt_factor=1.0, remove_small=50, fill_holes=None,
    #         clear_border=None, quant='mean')

    # root_dir = '/home/phuong/data/GEX/20210731/'
    # combine_before_after(root_dir)
    # compare_before_after(root_dir)
    # barplot_expts(root_dir)