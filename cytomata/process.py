import os
import warnings
import functools as ft

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from natsort import natsorted, ns
from skimage import img_as_float, img_as_uint, img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.measure import regionprops
from skimage.filters import gaussian, threshold_li, threshold_local
from skimage.morphology import remove_small_objects, erosion, disk
from skimage.restoration import denoise_nl_means, estimate_sigma
import FlowCal

from cytomata.utils import setup_dirs, list_img_files


def preprocess_img(imgf, offset=2):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    bkg = img.copy()
    sig = estimate_sigma(img)
    thr = threshold_li(bkg)
    thr = threshold_li(bkg[bkg < thr])
    bkg_sig = estimate_sigma(bkg[bkg < thr])
    c_bg = np.mean(bkg[bkg < thr]) - 1*np.std(bkg[bkg < thr])
    c_fg = np.mean(bkg[bkg > thr]) + 1*np.std(bkg[bkg > thr])
    cutoff = 100*c_bg/c_fg
    # print(cutoff)
    cutoff = 100 if cutoff > 60 else cutoff
    cutoff = max(min(cutoff, 100), 0)
    bkg[bkg >= np.percentile(bkg, cutoff)] = np.percentile(bkg, cutoff)
    bkg = gaussian(bkg, 100) + offset*bkg_sig
    bkg[bkg < 0] = 0
    img = (img - bkg)
    img[img < 0] = 0
    return img, raw, bkg


def reorganize_files(data_dir):
    for r, repeat_dir in enumerate(natsorted([x[1] for x in os.walk(data_dir)][0])):
        r_dir = os.path.join(data_dir, repeat_dir)
        os.rename(os.path.join(r_dir, 'u0.csv'), os.path.join(r_dir, 'u.csv'))
        os.rename(os.path.join(r_dir, 'mCherry', '0'), os.path.join(r_dir, 'imgs'))
        os.rmdir(os.path.join(r_dir, 'mCherry'))
        os.remove(os.path.join(r_dir, 'configs.txt'))
        os.remove(os.path.join(r_dir, 'settings.txt'))
        os.rename(r_dir, os.path.join(data_dir, str(r)))


def save_img_tiff(img, fname, save_dir):
    setup_dirs(os.path.join(save_dir, 'sub'))
    img_path = os.path.join(save_dir, 'sub', fname + '.tiff')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(img_path, img_as_uint(rescale_intensity(img)))


def split_mask(maskf):
    mask = img_as_float(imread(maskf))
    mask = remove_small_objects(ndi.label(mask)[0].astype(bool), min_size=10)
    labeled, n = ndi.label(mask)
    masks = [labeled == i for i in range(1, n+1)]
    centroids = {n: ndi.center_of_mass(mi) for n, mi in enumerate(masks)}
    regions = labeled > 0
    return regions, masks, centroids


def fnames_to_time(img_dir):
    img_files = list_img_files(img_dir)
    t = [float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in img_files]
    return t


def load_fc_data(fcs_path, channel):
    y = FlowCal.io.FCSData(fcs_path)
    y = FlowCal.transform.to_rfi(y)
    y = y[:, channel]
    return y


def calc_cmax(img_dir):
    img_files = list_img_files(img_dir)
    max_vals = [np.percentile(img_as_float(imread(imgf)), 100) for imgf in img_files]
    max_imgf = img_files[np.argmax(max_vals)]
    cmax = np.percentile(preprocess_img(max_imgf)[0], 100)
    return cmax


def calc_deltaf_f0(y_df):
    for n in y_df['n'].unique():
        yn = y_df.loc[(y_df['n'] == n), 'y']
        yn0 = y_df.loc[(y_df['n'] == n) & (y_df['t'] <= 10), 'y'].mean()
        y_df.loc[(y_df['n'] == n), 'y'] = (yn - yn0) / yn0
    return y_df


def calc_tl_diff(y_dfs):
    y_ave_dfs = []
    for y_df_i in y_dfs:
        y_df_i = y_df_i.drop(columns=['n'])
        y_df_i = y_df_i.groupby("t")['y'].mean()
        y_ave_dfs.append(y_df_i)
    y_ave_df = ft.reduce(lambda left, right: pd.merge(left, right, on='t'), y_ave_dfs)
    y_res_df = y_ave_df.sub(y_ave_df.iloc[:, 0], axis=0)
    y_res_df.columns = list(range(len(y_res_df.columns)))
    y_res_df = y_res_df.stack().reset_index()
    y_res_df.columns = ['t', 'h', 'y']
    return y_res_df


def calc_val_at_timepoints(y_dfs, timepoints):
    tp_df = pd.DataFrame()
    for h, y_df in enumerate(y_dfs):
        for tp in timepoints:
            tp_df_i = y_df.loc[(y_df['t'] == tp), ['y']]
            tp_df_i['h'] = np.ones_like(tp_df_i['y']) * h
            tp_df_i['t'] = np.ones_like(tp_df_i['y']) * tp
            tp_df = pd.concat([tp_df, tp_df_i], ignore_index=True)
    return tp_df


def process_u_csv(ty, u_csv, u_n, save_dir):
    setup_dirs(save_dir)
    t_on = []
    udf = pd.read_csv(u_csv)
    ty = np.around(ty, 0)
    tu = np.around(np.arange(ty[0], ty[-1] + 1, 1), 0)
    uta = np.ceil(udf['ta'].values)
    utb = np.floor(udf['tb'].values)
    u = np.zeros_like(tu)
    for ta, tb in zip(uta, utb):
        if ta > ty[-1]:
            continue
        t_on += list(np.arange(round(ta, 0), round(tb, 0) + 0.1, 1))
        ia = list(tu).index(ta)
        ib = list(tu).index(tb)
        u[ia:ib+1] = 1
    t_ann_img = []
    for tbl in t_on:
        t_ann_img.append(min(ty, key=lambda ti: abs(ti - tbl)))
    t_ann_img = np.unique(t_ann_img)
    n = np.ones_like(tu) * u_n
    u_df = pd.DataFrame({'t': tu, 'u': u, 'n': n})
    return u_df, t_ann_img


def revert_tu(u_df):
    u_ave = u_df.groupby('t', as_index=False)['u'].mean()
    u_ave_01 = (u_ave['u'] > 0).astype(int)
    u_diff = np.diff(u_ave_01)
    ta_idx = u_ave['t'][np.flatnonzero(u_diff > 0)]
    tb_idx = u_ave['t'][np.flatnonzero(u_diff < 0)]
    ta_tb = np.column_stack((ta_idx, tb_idx))
    return ta_tb


def process_fit_data(data_dir):
    y_csv = os.path.join(data_dir, 'y.csv')
    u_csv = os.path.join(data_dir, 'u.csv')
    y_df = pd.read_csv(y_csv)
    u_df = pd.read_csv(u_csv)
    y_df = y_df.groupby('t')['y'].agg(['mean', 'sem'])
    td = y_df.index.values
    yd = y_df['mean'].values
    sem = y_df['sem'].values
    u_df = u_df.groupby('t')['u'].agg(['mean'])
    tu = u_df.index.values
    ud = u_df['mean'].values
    ud[ud > 0] = 1
    ud[ud < 0] = 0
    uf = interp1d(tu, ud, kind='linear')
    return td, yd, sem, tu, ud, uf


def approx_half_life(t, y, n):
    i_exc = np.argmax(np.abs(y[:18]))
    t_exc = t[i_exc]
    y_exc = y[i_exc]
    y_bas = np.mean(y[:i_exc-5])
    tp = t[i_exc:]
    tp = tp - np.min(tp)
    yp = y[i_exc:]
    yf = interp1d(tp, yp, 'linear')
    ti = np.arange(tp[0], tp[-1], 1)
    yi = yf(ti)
    y_half = (y_bas + y_exc)/2
    idx = np.argmin(np.abs((yi - y_half)))
    t_half = np.around(ti[idx], 0)
    return t_half


def calc_t_half_data(y_dfs):
    t_half_vals = []
    h_vals = []
    for h, y_df_h in enumerate(y_dfs):
        for n in y_df_h['n'].unique():
            try:
                y_df_hn = y_df_h.loc[(y_df_h['n'] == n)]
                t_vals = y_df_hn['t'].values
                y_vals = y_df_hn['y'].values
                t_half_val = approx_half_life(t_vals, y_vals, n)
                t_half_vals.append(t_half_val)
                h_vals.append(h)
            except Exception:
                pass
    t_half_df = pd.DataFrame({'x': h_vals, 'y': t_half_vals})
    return t_half_df


def rescale_lov(y):
    ymax = np.mean(y[:60])
    ymin = np.min(y)
    return (y - ymin)/(ymax - ymin)


def rescale_ilid(y):
    ymax = np.max(y)
    ymin = np.mean(y[:60])
    return (y - ymin)/(ymax - ymin)


def rescale_01(y):
    return (y-min(y))/(max(y) - min(y))
