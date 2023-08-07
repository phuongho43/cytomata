import os
import sys
import time
import itertools
from joblib import Parallel, delayed
sys.path.append(os.path.abspath('../'))

import numpy as np
import scipy as sp
import pingouin as pg
import pandas as pd
from natsort import natsorted, ns
from tabulate import tabulate


from cytomata.plot import plot_cell_img, plot_bkg_profile, plot_class_group, plot_uy, plot_fc_hist
from cytomata.process import (preprocess_img, reorganize_files, save_img_tiff, split_mask,
    process_u_csv, fnames_to_time, calc_cmax, revert_tu, load_fc_data, calc_deltaf_f0,
    calc_val_at_timepoints, calc_t_half_data)
from cytomata.utils import setup_dirs, list_img_files, custom_styles, custom_palette


def process_gene_expr(root_dir):
    def img_task(yi, imgf):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg = preprocess_img(imgf, offset=5)
        plot_bkg_profile(raw, bkg, fname, save_dir)
        plot_cell_img(img, fname, os.path.join(save_dir, 'img'), cmax=cmax, sb_microns=330)
        return np.mean(img)
    ta = time.time()
    data = []
    data_dir = os.path.join(root_dir, 'data')
    for c, class_dir in enumerate(natsorted([x[1] for x in os.walk(data_dir)][0])):
        c_dir = os.path.join(data_dir, class_dir)
        for r, repeat_dir in enumerate(natsorted([x[1] for x in os.walk(c_dir)][0])):
            r_dir = os.path.join(c_dir, repeat_dir)
            for g, group_dir in enumerate(natsorted([x[1] for x in os.walk(r_dir)][0])):
                save_dir = os.path.join(root_dir, 'results', class_dir, repeat_dir, group_dir)
                g_dir = os.path.join(r_dir, group_dir)
                cmax = calc_cmax(g_dir)
                yi = []
                yi = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(yi, imgf) for imgf in list_img_files(g_dir))
                yi = np.mean(yi)
                if g == 0:
                    y0 = yi
                yi = yi
                print(tabulate([
                    ['elapsed_time', time.time() - ta],
                    ['class', class_dir],
                    ['repeat', repeat_dir],
                    ['group', group_dir],
                    ['response', yi],
                ]))
                data.append({'repeat': r, 'class': c, 'group': g, 'response': yi})
    y_df = pd.DataFrame(data)
    y_df.to_csv(os.path.join(root_dir, 'results', 'y.csv'), index=False)
    return y_df


def process_biosensor_timelapse(root_dir):
    def img_task(yi, imgf, maskf, n_max):
        img, raw, bkg = preprocess_img(imgf, offset=2)
        fname = os.path.splitext(os.path.basename(imgf))[0]
        plot_bkg_profile(raw, bkg, fname, save_dir)
        save_img_tiff(img, fname, save_dir)
        plot_cell_img(img, fname, os.path.join(save_dir, 'img'), cmax=cmax, sb_microns=11)
        regions, masks, centroids = split_mask(maskf)
        plot_cell_img(img, fname, os.path.join(save_dir, 'roi'), regions=regions, centroids=centroids, cmax=cmax, sb_microns=11)
        rows = [{'t': np.around(float(fname), 0), 'y': np.mean(img[mask]), 'n': n_max + n} for n, mask in enumerate(masks)]
        return rows
    ta = time.time()
    data_dir = os.path.join(root_dir, 'data')
    y_df = pd.DataFrame()
    u_df = pd.DataFrame()
    n_max = 0
    for r, repeat_dir in enumerate(natsorted([x[1] for x in os.walk(data_dir)][0])):
        r_dir = os.path.join(data_dir, repeat_dir)
        img_dir = os.path.join(r_dir, 'imgs')
        save_dir = os.path.join(root_dir, 'results', repeat_dir)
        cmax = calc_cmax(img_dir)
        maskf = os.path.join(r_dir, 'Mask.tif')
        u_csv = os.path.join(r_dir, 'u.csv')
        ty = fnames_to_time(img_dir)
        u_df_i, t_ann_img = process_u_csv(ty, u_csv, r, save_dir)
        yi = []
        yi = Parallel(n_jobs=os.cpu_count()-2)(delayed(img_task)(yi, imgf, maskf, n_max) for imgf in list_img_files(img_dir))
        yi = list(itertools.chain.from_iterable(yi))
        y_df_i = pd.DataFrame(yi)
        y_df_i = calc_deltaf_f0(y_df_i)
        y_df_i.to_csv(os.path.join(save_dir, 'y.csv'), index=False)
        u_df_i.to_csv(os.path.join(save_dir, 'u.csv'), index=False)
        y_df = pd.concat([y_df, y_df_i])
        u_df = pd.concat([u_df, u_df_i])
        n_max = y_df_i.n.max()
        y_df_i.n = y_df_i.n - y_df_i.n.min()
        plot_uy(y_df_i, u_df_i, save_dir, dpi=100)
        for n in y_df_i['n'].unique():
            yni_df = y_df_i.loc[(y_df_i['n'] == n)]
            fname = str(n) + '.png'
            plot_uy(yni_df, u_df_i, save_dir=os.path.join(save_dir, 'plot'), fname=fname, dpi=100)
        print(tabulate([
            ['elapsed_time', time.time() - ta],
            ['repeat', repeat_dir],
            ['num_cells', n_max + 1],
        ]))
    y_df.to_csv(os.path.join(root_dir, 'results', 'y.csv'), index=False)
    u_df.to_csv(os.path.join(root_dir, 'results', 'u.csv'), index=False)
    plot_uy(y_df, u_df, os.path.join(root_dir, 'results'), ymin=None, ymax=None)


def process_fc_data(root_dir):
    y_df = pd.DataFrame()
    data_dir = os.path.join(root_dir, 'data')
    for c, class_dir in enumerate(natsorted([x[1] for x in os.walk(data_dir)][0])):
        c_dir = os.path.join(data_dir, class_dir)
        for r, repeat_dir in enumerate(natsorted([x[1] for x in os.walk(c_dir)][0])):
            r_dir = os.path.join(c_dir, repeat_dir)
            for g, fcs_fn in enumerate(natsorted([x[2] for x in os.walk(r_dir)][0])):
                fcs_i = os.path.join(r_dir, fcs_fn)
                y_col = load_fc_data(fcs_i, 'FL4-A')
                r_col = np.ones_like(y_col) * r
                c_col = np.ones_like(y_col) * c
                g_col = np.ones_like(y_col) * g
                y_df_i = pd.DataFrame({'repeat': r_col, 'class': c_col, 'group': g_col, 'response': y_col})
                y_df = pd.concat([y_df, y_df_i])
    y_df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)
    return y_df


def plot_figure_1():
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/0-ddFP/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/1-LOV/I427V/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/1-LOV/V416I/results/y.csv')
    y_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/2-iLID/I427V/results/y.csv')
    y_df4 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/2-iLID/V416I/results/y.csv')
    tp_df = calc_val_at_timepoints([y_df0, y_df1, y_df2, y_df3, y_df4], [60])
    ddfp = tp_df.loc[(tp_df['h']==0), 'y']
    lovf = tp_df.loc[(tp_df['h']==1), 'y']
    ilidf = tp_df.loc[(tp_df['h']==3), 'y']
    ilids = tp_df.loc[(tp_df['h']==4), 'y']
    print(sp.stats.ttest_ind(ddfp, lovf))
    print(sp.stats.ttest_ind(ddfp, ilidf))
    print(sp.stats.ttest_ind(ilidf, ilids))
    save_path = '/home/phuong/data/1-fakr/1-ddFP/0-expression/1-100x/t60.png'
    palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#29CA6E']
    group_labels = ['ddFP', 'LOVfast', 'LOVslow', 'iLIDfast', 'iLIDslow']
    class_labels = []
    ylabel = 'AU'
    plot_class_group(tp_df, save_path, group_labels, class_labels=[],
        x_var='h', y_var='y', h_var=None, ylabel=ylabel, xlabel='',
        ymin=None, ymax=None, palette=palette, figsize=(24, 16))


def plot_figure_2():
    group_labels = ['ddFP', 'LOVfast', 'LOVslow']
    palette = ['#34495E', '#785EF0', '#FE6100']
    order = [0, 1, 2]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/'
    u_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/0-ddFP/results/u.csv')
    u_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/results/u.csv')
    u_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/V416I/results/u.csv')
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/0-ddFP/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/V416I/results/y.csv')
    u_df = pd.concat([u_df0, u_df1, u_df2])
    for h, y_df_i in enumerate([y_df0, y_df1, y_df2]):
        y_df_i['h'] = np.ones_like(y_df_i.y) * h
    y_df = pd.concat([y_df0, y_df1, y_df2])
    plot_uy(y_df, u_df, save_dir, lgd_loc='lower right', ylabel=r'$\mathbf{\Delta F/F_{0}}$',
        ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=order, palette=palette)


def plot_figure_3():
    group_labels = ['ddFP', 'iLIDfast', 'iLIDslow']
    palette = ['#34495E', '#785EF0', '#FE6100']
    order = [0, 1, 2]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/'
    u_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/0-ddFP/results/u.csv')
    u_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/u.csv')
    u_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/V416I/results/u.csv')
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/0-ddFP/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/V416I/results/y.csv')
    u_df = pd.concat([u_df0, u_df1, u_df2])
    for h, y_df_i in enumerate([y_df0, y_df1, y_df2]):
        y_df_i['h'] = np.ones_like(y_df_i.y) * h
    y_df = pd.concat([y_df0, y_df1, y_df2])
    plot_uy(y_df, u_df, save_dir, lgd_loc='upper right', ylabel=r'$\mathbf{\Delta F/F_{0}}$',
        ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=order, palette=palette)


def plot_figure_4():
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/0-ddFP/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/V416I/results/y.csv')
    y_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/y.csv')
    y_df4 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/V416I/results/y.csv')
    tp_df = calc_val_at_timepoints([y_df0, y_df1, y_df2, y_df3, y_df4], [60, 62])
    ddFP60 = tp_df.loc[(tp_df['h']==0) & (tp_df['t']==60), 'y']
    ddFP62 = tp_df.loc[(tp_df['h']==0) & (tp_df['t']==62), 'y']
    LOVf60 = tp_df.loc[(tp_df['h']==1) & (tp_df['t']==60), 'y']
    LOVf62 = tp_df.loc[(tp_df['h']==1) & (tp_df['t']==62), 'y']
    LOVs60 = tp_df.loc[(tp_df['h']==2) & (tp_df['t']==60), 'y']
    LOVs62 = tp_df.loc[(tp_df['h']==2) & (tp_df['t']==62), 'y']
    iLIDf60 = tp_df.loc[(tp_df['h']==3) & (tp_df['t']==60), 'y']
    iLIDf62 = tp_df.loc[(tp_df['h']==3) & (tp_df['t']==62), 'y']
    iLIDs60 = tp_df.loc[(tp_df['h']==4) & (tp_df['t']==60), 'y']
    iLIDs62 = tp_df.loc[(tp_df['h']==4) & (tp_df['t']==62), 'y']
    print(sp.stats.ttest_ind(ddFP60, ddFP62))
    print(sp.stats.ttest_ind(LOVf60, LOVf62))
    print(sp.stats.ttest_ind(LOVs60, LOVs62))
    print(sp.stats.ttest_ind(iLIDf60, iLIDf62))
    print(sp.stats.ttest_ind(iLIDs60, iLIDs62))
    save_path = '/home/phuong/data/1-fakr/1-ddFP/4-training/t60t62.png'
    palette = ['#34495E', '#648FFF']
    rc = {'axes.labelsize': 72, 'xtick.labelsize': 64, 'ytick.labelsize': 64, 'legend.fontsize': 56}
    group_labels = ['ddFP', 'LOVfast', 'LOVslow', 'iLIDfast', 'iLIDslow']
    class_labels = ['Before BL', 'After BL']
    plot_class_group(tp_df, save_path, group_labels, class_labels=class_labels,
        x_var='h', y_var='y', h_var='t', ylabel=r'$\mathbf{\Delta F/F_{0}}$', xlabel='',
        ymin=None, ymax=None, lgd_loc='best', palette=palette, figsize=(24, 16), rc=rc)


def plot_figure_5():
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/V416I/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/y.csv')
    y_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/V416I/results/y.csv')
    group_labels = ['LOVfast', 'LOVslow', 'iLIDfast', 'iLIDslow']
    ylabel = r'$\mathbf{t_{1/2}\ (s)}$'
    palette = ['#648FFF', '#785EF0', '#DC267F', '#FE6100']
    save_path = '/home/phuong/data/1-fakr/1-ddFP/4-training/t_half.png'
    t_half_df = calc_t_half_data([y_df0, y_df1, y_df2, y_df3])
    t_half_df.to_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/t_half.csv', index=False)
    t_half_df = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/4-training/t_half.csv')
    LOVf = t_half_df.loc[(t_half_df['x']==0), 'y']
    LOVs = t_half_df.loc[(t_half_df['x']==1), 'y']
    iLIDf = t_half_df.loc[(t_half_df['x']==2), 'y']
    iLIDs = t_half_df.loc[(t_half_df['x']==3), 'y']
    print("LOVfast Mean", np.mean(LOVf))
    print("LOVslow Mean", np.mean(LOVs))
    print("iLIDfast Mean", np.mean(iLIDf))
    print("LOVslow Mean", np.mean(iLIDs))
    print('T-Test:')
    print(sp.stats.ttest_ind(LOVf, LOVs))
    print(sp.stats.ttest_ind(iLIDf, iLIDs))
    print(sp.stats.ttest_ind(LOVf, iLIDf))
    print(sp.stats.ttest_ind(LOVs, iLIDs))
    plot_class_group(t_half_df, save_path, group_labels, class_labels=[],
        x_var='x', y_var='y', h_var=None, ylabel=ylabel, xlabel='',
        ymin=None, ymax=None, palette=palette, figsize=(24, 16))


def plot_figure_6():
    group_labels = [r'$2000\ uW/mm^2$', r'$200\ uW/mm^2$', r'$20\ uW/mm^2$']
    palette = ['#785EF0', '#DC267F', '#FE6100']
    order = [2, 1, 0]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/1-intensity/'
    u_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL20uW/results/u.csv')
    u_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL200uW/results/u.csv')
    u_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL2000uW/results/u.csv')
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL20uW/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL200uW/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/1-intensity/BL2000uW/results/y.csv')
    u_df = pd.concat([u_df0, u_df1, u_df2])
    for h, y_df_i in enumerate([y_df0, y_df1, y_df2]):
        y_df_i['h'] = np.ones_like(y_df_i.y) * h
    y_df = pd.concat([y_df0, y_df1, y_df2])
    plot_uy(y_df, u_df, save_dir, lgd_loc='lower right', ylabel=r'$\mathbf{\Delta F/F_{0}}$',
        ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=order, palette=palette)
    tp_df = calc_val_at_timepoints([y_df0, y_df1, y_df2], [60, 62])
    BL20 = tp_df.loc[(tp_df['h']==0) & (tp_df['t']==62), 'y']
    BL200 = tp_df.loc[(tp_df['h']==1) & (tp_df['t']==62), 'y']
    BL2000 = tp_df.loc[(tp_df['h']==2) & (tp_df['t']==62), 'y']
    print(sp.stats.ttest_ind(BL20, BL200))
    print(sp.stats.ttest_ind(BL200, BL2000))
    save_path = '/home/phuong/data/1-fakr/1-ddFP/1-intensity/t60t62.png'
    group_labels = [r'$\mathdefault{20\ uW/mm^2}$', r'$\mathdefault{200\ uW/mm^2}$', r'$\mathdefault{2000\ uW/mm^2}$']
    palette = ['#34495E', '#648FFF']
    class_labels = ['Before BL', 'After BL']
    plot_class_group(tp_df, save_path, group_labels, class_labels=class_labels,
        x_var='h', y_var='y', h_var='t', ylabel=r'$\mathbf{\Delta F/F_{0}}$', xlabel='',
        ymin=None, ymax=None, lgd_loc='best', palette=palette, figsize=(24, 16))


def plot_figure_7():
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/0-13AA/I427V/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/1-20AA/I427V/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/0-13AA/V416I/results/y.csv')
    y_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/1-20AA/V416I/results/y.csv')
    u_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/0-13AA/I427V/results/u.csv')
    u_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/1-20AA/I427V/results/u.csv')
    u_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/0-13AA/V416I/results/u.csv')
    u_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/2-linker/1-20AA/V416I/results/u.csv')
    u_df = pd.concat([u_df0, u_df1, u_df2, u_df3])
    for h, y_df_i in enumerate([y_df0, y_df1, y_df2, y_df3]):
        y_df_i['h'] = np.ones_like(y_df_i.y) * h
    y_df = pd.concat([y_df0, y_df1, y_df2, y_df3])
    group_labels = ['iLIDfast (13AA)', 'iLIDfast (20AA)', 'iLIDslow (13AA)', 'iLIDslow (20AA)']
    palette = ['#785EF0', '#DC267F', '#FE6100', '#29CA6E']
    order = None
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/2-linker/'
    plot_uy(y_df, u_df, save_dir, lgd_loc='best', ylabel=r'$\mathbf{\Delta F/F_{0}}$',
        ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=order, palette=palette)
    tp_df = calc_val_at_timepoints([y_df0, y_df1, y_df2, y_df3], [62]).drop(columns=['t'])
    AA13f = tp_df.loc[(tp_df['h']==0), 'y']
    AA20f = tp_df.loc[(tp_df['h']==1), 'y']
    AA13s = tp_df.loc[(tp_df['h']==2), 'y']
    AA20s = tp_df.loc[(tp_df['h']==3), 'y']
    print(sp.stats.ttest_ind(AA13f, AA20f))
    print(sp.stats.ttest_ind(AA13s, AA20s))
    tp_df.loc[(tp_df['h'] == 0) | (tp_df['h'] == 2), 'n'] = 0
    tp_df.loc[(tp_df['h'] == 1) | (tp_df['h'] == 3), 'n'] = 1
    tp_df['h'] = tp_df['h'].replace({1: 0, 2: 1, 3: 1})
    save_path = '/home/phuong/data/1-fakr/1-ddFP/2-linker/t62.png'
    palette = ['#648FFF', '#785EF0']
    group_labels = ['iLIDfast', 'iLIDslow']
    class_labels = ['13AA Linker', '20AA Linker']
    rc = {'axes.labelsize': 72, 'xtick.labelsize': 64, 'ytick.labelsize': 64, 'legend.fontsize': 56}
    plot_class_group(tp_df, save_path, group_labels, class_labels=class_labels,
        x_var='h', y_var='y', h_var='n', ylabel=r'$\mathbf{\Delta F/F_{0}}$', xlabel='',
        ymin=None, ymax=None, lgd_loc='best', palette=palette, figsize=(24, 16))


def plot_figure_8():
    y_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/0-cyto/1-LOVfast/results/y.csv')
    y_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/1-nucl/1-LOVfast/results/y.csv')
    y_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/0-cyto/2-iLIDfast/results/y.csv')
    y_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/1-nucl/2-iLIDfast/results/y.csv')
    u_df0 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/0-cyto/1-LOVfast/results/u.csv')
    u_df1 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/1-nucl/1-LOVfast/results/u.csv')
    u_df2 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/0-cyto/2-iLIDfast/results/u.csv')
    u_df3 = pd.read_csv('/home/phuong/data/1-fakr/1-ddFP/3-subloc/1-nucl/2-iLIDfast/results/u.csv')
    u_df = pd.concat([u_df0, u_df1, u_df2, u_df3])
    for h, y_df_i in enumerate([y_df0, y_df1, y_df2, y_df3]):
        y_df_i['h'] = np.ones_like(y_df_i.y) * h
    y_df = pd.concat([y_df0, y_df1, y_df2, y_df3])
    group_labels = ['LOVfast (cyto)', 'LOVfast (nucl)', 'iLIDfast (cyto)', 'iLIDfast (nucl)']
    palette = ['#785EF0', '#DC267F', '#FE6100', '#29CA6E']
    order = None
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/3-subloc/'
    plot_uy(y_df, u_df, save_dir, lgd_loc='upper right', ylabel=r'$\mathbf{\Delta F/F_{0}}$',
        ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=order, palette=palette)
    tp_df = calc_val_at_timepoints([y_df0, y_df1, y_df2, y_df3], [62]).drop(columns=['t'])
    LOVc = tp_df.loc[(tp_df['h']==0), 'y']
    LOVn = tp_df.loc[(tp_df['h']==1), 'y']
    iLIDc = tp_df.loc[(tp_df['h']==2), 'y']
    iLIDn = tp_df.loc[(tp_df['h']==3), 'y']
    print(sp.stats.ttest_ind(LOVc, LOVn))
    print(sp.stats.ttest_ind(iLIDc, iLIDn))
    tp_df.loc[(tp_df['h'] == 0) | (tp_df['h'] == 2), 'n'] = 0
    tp_df.loc[(tp_df['h'] == 1) | (tp_df['h'] == 3), 'n'] = 1
    tp_df['h'] = tp_df['h'].replace({1: 0, 2: 1, 3: 1})
    save_path = '/home/phuong/data/1-fakr/1-ddFP/3-subloc/t62.png'
    palette = ['#648FFF', '#785EF0']
    group_labels = ['LOVfast', 'iLIDfast']
    class_labels = ['cytoplasmic', 'nuclear']
    rc = {'axes.labelsize': 72, 'xtick.labelsize': 64, 'ytick.labelsize': 64, 'legend.fontsize': 56}
    plot_class_group(tp_df, save_path, group_labels, class_labels=class_labels,
        x_var='h', y_var='y', h_var='n', ylabel=r'$\mathbf{\Delta F/F_{0}}$', xlabel='',
        ymin=None, ymax=None, lgd_loc='best', palette=palette, figsize=(24, 16))


def plot_figure_10():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/1--basal/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    TetO_mean = y_df.loc[(y_df['group']==0), 'response'].mean()
    for g in y_df['group'].unique():
        y_df.loc[(y_df['group']==g), 'response'] = y_df.loc[(y_df['group']==g), 'response']/TetO_mean
    y_df['response'] = np.log2(y_df['response'])
    TetO = y_df.loc[(y_df['group']==0), 'response']
    iLIDf = y_df.loc[(y_df['group']==1), 'response']
    iLIDs = y_df.loc[(y_df['group']==2), 'response']
    LOVf = y_df.loc[(y_df['group']==3), 'response']
    sparse = y_df.loc[(y_df['group']==4), 'response']
    print(sp.stats.ttest_ind(TetO, iLIDf))
    print(sp.stats.ttest_ind(TetO, iLIDs))
    print(sp.stats.ttest_ind(TetO, LOVf))
    print(sp.stats.ttest_ind(TetO, sparse))
    class_labels = []
    group_labels = ['TetO', 'Dense\nChannel', 'iLIDslow', 'LOVfast', 'Sparse\nChannel']
    palette = ['#34495E', '#785EF0', '#FE6100', '#648FFF', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    ylabel = r'$\mathdefault{Log_2\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=[],
        x_var='group', y_var='response', h_var=None, ylabel=ylabel, xlabel='',
        ymin=None, ymax=None, palette=palette, figsize=(24, 16))


def plot_figure_11():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/2--BL-intensity/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    rep5hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response']
    dense5hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    dense10hz = y_df.loc[(y_df['group']==3) & (y_df['class']==1), 'response']
    dense50hz = y_df.loc[(y_df['group']==4) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(rep5hz, dense5hz))
    print(sp.stats.ttest_ind(dense5hz, dense10hz))
    print(sp.stats.ttest_ind(dense10hz, dense50hz))
    group_labels = ['0', '1', '5', '10', '50']
    class_labels = ['Reporter Only', 'Dense Channel']
    palette = ['#34495E', '#785EF0']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Intensity\ \mu W/mm^2}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_12():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/3--iLID-freq/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    ilidf025hz = y_df.loc[(y_df['group']==3) & (y_df['class']==0), 'response']
    ilids025hz = y_df.loc[(y_df['group']==3) & (y_df['class']==1), 'response']
    ilidf1hz = y_df.loc[(y_df['group']==5) & (y_df['class']==0), 'response']
    ilids1hz = y_df.loc[(y_df['group']==5) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(ilidf025hz, ilids025hz))
    print(sp.stats.ttest_ind(ilidf1hz, ilids1hz))
    group_labels = ['0', '0.05', '0.1', '0.25', '0.5', '1']
    class_labels = ['TetR-iLIDfast\n+ sspB-VP64', 'TetR-iLIDslow\n+ sspB-VP64']
    palette = ['#785EF0', '#FE6100']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_13():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/4--dense-sparse/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    dense025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response']
    sparse025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response']
    dense1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response']
    sparse1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(dense025hz, sparse025hz))
    print(sp.stats.ttest_ind(dense1hz, sparse1hz))
    group_labels = ['0', '0.05', '0.1', '0.25', '0.5', '1']
    class_labels = ['Dense Channel', 'Sparse Channel']
    palette = ['#785EF0', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_14():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/5--repression/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    ilid_0hz = y_df.loc[(y_df['group']==0) & (y_df['class']==1), 'response']
    ilid_025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response']
    ilid_1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    lov_0hz = y_df.loc[(y_df['group']==0) & (y_df['class']==0), 'response']
    lov_025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response']
    lov_1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response']
    sparse_025hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    sparse_1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==2), 'response']
    print(sp.stats.ttest_ind(lov_025hz, lov_1hz))
    print(sp.stats.ttest_ind(sparse_025hz, sparse_1hz))
    print(sp.stats.ttest_ind(ilid_1hz, sparse_1hz))
    group_labels = ['0', '0.25', '1']
    class_labels = ['TetR-LOVfast + Zdk-VP64', 'TetR-iLIDslow + sspB-VP64', 'Sparse Channel']
    palette = ['#648FFF', '#FE6100', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='lower left', palette=palette, figsize=(24, 16))


def plot_figure_15():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/6--dual-ch/0--12TetO-YB-mGd_TetR-mNLS-iLID27V_sspBn-mNLS-VP64_12UAS-YB-mScI_Gal4-mNLS-LOV27V_Zdk1-mNLS-iLID16I/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    dense025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response']
    sparse025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response']
    dense1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response']
    sparse1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(dense025hz, sparse025hz))
    print(sp.stats.ttest_ind(dense1hz, sparse1hz))
    group_labels = ['0', '0.25', '1']
    class_labels = ['Dual Channel\n(Dense Branch)', 'Dual Channel\n(Sparse Branch)']
    palette = ['#785EF0', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_16():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/6--dual-ch/1--8ZF3s-YB-mGd_ZF3-mNLS-iLID27V_sspBn-mNLS-p65_8ZF1s-YB-mScI_ZF1-mNLS-LOV27V_Zdk1-mNLS-iLID16I/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    dense025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response']
    sparse025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response']
    dense1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response']
    sparse1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(dense025hz, sparse025hz))
    print(sp.stats.ttest_ind(dense1hz, sparse1hz))
    group_labels = ['0', '0.25', '1']
    class_labels = ['Dual Channel\n(Dense Branch)', 'Dual Channel\n(Sparse Branch)']
    palette = ['#785EF0', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_17():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/7--K562-freq/'
    y_df = pd.read_csv(os.path.join(root_dir, 'results', 'y.csv'))
    y_df['response'] = np.log2(y_df['response'])
    dense025hz = y_df.loc[(y_df['group']==3) & (y_df['class']==0), 'response']
    sparse025hz = y_df.loc[(y_df['group']==3) & (y_df['class']==1), 'response']
    dense1hz = y_df.loc[(y_df['group']==5) & (y_df['class']==0), 'response']
    sparse1hz = y_df.loc[(y_df['group']==5) & (y_df['class']==1), 'response']
    print(sp.stats.ttest_ind(dense025hz, sparse025hz))
    print(sp.stats.ttest_ind(dense1hz, sparse1hz))
    group_labels = ['0', '0.05', '0.1', '0.25', '0.5', '1']
    class_labels = ['Dense Channel\n(TetO-RFP)', 'Sparse Channel\n(TetO-RFP)']
    palette = ['#785EF0', '#DC267F']
    save_path = os.path.join(root_dir, 'results', 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


def plot_figure_18():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/8--fc-staining/0--int/'
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    y_df = y_df.loc[(y_df['class'] == 0) & (y_df['response'] > 0)]
    group_labels = [
        r'$\mathdefault{Plain\ K562}$',
        r'$\mathdefault{Constitutive}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ Dark}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 5\ uW/mm^2}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 10\ uW/mm^2}$'
    ]
    palette = ['#648FFF', '#DC267F', '#FE6100']
    xlabel = 'anti-hCD19 AF647 (AU)'
    save_path = os.path.join(root_dir, 'y_CD19.png')
    gateline=3e3
    plot_fc_hist(y_df, gateline, save_path, group_labels, xlabel, palette=palette, figsize=(32, 16))


def plot_figure_19():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/8--fc-staining/0--int/'
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    y_df = y_df.loc[(y_df['class'] == 1) & (y_df['response'] > 0)]
    group_labels = [
        r'$\mathdefault{Plain\ K562}$',
        r'$\mathdefault{Constitutive}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ Dark}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 5\ uW/mm^2}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 10\ uW/mm^2}$'
    ]
    palette = ['#648FFF', '#DC267F', '#FE6100']
    xlabel = 'anti-hPSMA APC (AU)'
    save_path = os.path.join(root_dir, 'y_PSMA.png')
    gateline=3e3
    plot_fc_hist(y_df, gateline, save_path, group_labels, xlabel, palette=palette, figsize=(32, 16))


def plot_figure_20():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/8--fc-staining/1--freq'
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    y_df = y_df.loc[(y_df['class'] == 0) & (y_df['response'] > 0)]
    group_labels = [
        r'$\mathdefault{Plain\ K562}$',
        r'$\mathdefault{Constitutive}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ Dark}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 0.25\ Hz}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 1.00\ Hz}$'
    ]
    palette = ['#648FFF', '#DC267F', '#FE6100']
    xlabel = 'anti-hCD19 AF647 (AU)'
    save_path = os.path.join(root_dir, 'y_CD19.png')
    gateline=3e3
    plot_fc_hist(y_df, gateline, save_path, group_labels, xlabel, palette=palette, figsize=(32, 16))


def plot_figure_21():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/8--fc-staining/1--freq/'
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    y_df = y_df.loc[(y_df['class'] == 1) & (y_df['response'] > 0)]
    group_labels = [
        r'$\mathdefault{Plain\ K562}$',
        r'$\mathdefault{Constitutive}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ Dark}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 0.25\ Hz}$',
        r'$\mathdefault{Inducible\ \mathrm{-}\ 1.00\ Hz}$'
    ]
    palette = ['#648FFF', '#DC267F', '#FE6100']
    xlabel = 'anti-hPSMA APC (AU)'
    save_path = os.path.join(root_dir, 'y_PSMA.png')
    gateline=3e3
    plot_fc_hist(y_df, gateline, save_path, group_labels, xlabel, palette=palette, figsize=(32, 16))


def plot_figure_22():
    root_dir = '/home/phuong/data/1-fakr/2-transcription/9--killing-assay/'
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    dense_ctr = y_df.loc[(y_df['group']==0) & (y_df['class']==0), 'response'].values
    dense0hz = np.log2(dense_ctr/dense_ctr)
    dense025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response'].values
    dense025hz = np.log2(dense025hz/dense_ctr)
    dense1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response'].values
    dense1hz = np.log2(dense1hz/dense_ctr)
    sparse_ctr = y_df.loc[(y_df['group']==0) & (y_df['class']==1), 'response'].values
    sparse0hz = np.log2(sparse_ctr/sparse_ctr)
    sparse025hz = y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response'].values
    sparse025hz = np.log2(sparse025hz/sparse_ctr)
    sparse1hz = y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response'].values
    sparse1hz = np.log2(sparse1hz/sparse_ctr)
    print(sp.stats.ttest_ind(dense025hz, sparse025hz))
    print(sp.stats.ttest_ind(dense1hz, sparse1hz))
    y_df.loc[(y_df['group']==0) & (y_df['class']==0), 'response'] = dense0hz
    y_df.loc[(y_df['group']==1) & (y_df['class']==0), 'response'] = dense025hz
    y_df.loc[(y_df['group']==2) & (y_df['class']==0), 'response'] = dense1hz
    y_df.loc[(y_df['group']==0) & (y_df['class']==1), 'response'] = sparse0hz
    y_df.loc[(y_df['group']==1) & (y_df['class']==1), 'response'] = sparse025hz
    y_df.loc[(y_df['group']==2) & (y_df['class']==1), 'response'] = sparse1hz
    group_labels = ['0', '0.25', '1']
    class_labels = ['Dense Channel\n(TetO-hCD19)', 'Sparse Channel\n(TetO-hPSMA)']
    palette = ['#785EF0', '#DC267F']
    save_path = os.path.join(root_dir, 'y.png')
    xlabel = r'$\mathdefault{BL\ Pulsing\ Frequency\ (Hz)}$'
    ylabel = r'$\mathdefault{Log_2\ Fold\ Ratio}$'
    plot_class_group(y_df, save_path, group_labels, class_labels=class_labels,
        x_var='group', y_var='response', h_var='class', ylabel=ylabel, xlabel=xlabel,
        ymin=None, ymax=None, lgd_loc='upper left', palette=palette, figsize=(24, 16))


if __name__ == '__main__':
    # process_gene_expr('/home/phuong/data/1-fakr/2-transcription/1--basal/')

    # reorganize_files('/home/phuong/data/1-fakr/1-ddFP/5-validation/data2/')

    # process_biosensor_timelapse('/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/')
    # print('\a')

    # root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    # y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    # u_df = pd.read_csv(os.path.join(root_dir, 'u.csv'))
    # palette = ['#DC267F']
    # plot_uy(y_df, u_df, os.path.join(root_dir), overlay_tu=True, ymin=None, ymax=None, palette=palette)

    # root_dir = '/home/phuong/data/1-fakr/2-transcription/8--fc-staining/0--int/'
    # process_fc_data(root_dir)


    # plot_figure_1()
    # plot_figure_2()
    # plot_figure_3()
    # plot_figure_4()
    # plot_figure_5()
    # plot_figure_6()
    # plot_figure_7()
    # plot_figure_8()

    # plot_figure_10()
    # plot_figure_11()
    # plot_figure_12()
    # plot_figure_13()
    # plot_figure_14()
    # plot_figure_15()
    # plot_figure_16()
    # plot_figure_17()
    # plot_figure_18()
    # plot_figure_19()
    # plot_figure_20()
    # plot_figure_21()
    plot_figure_22()

