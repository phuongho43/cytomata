import os
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import interp1d
from skimage import img_as_float, img_as_uint
from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_li, threshold_local
from skimage.morphology import remove_small_objects, erosion, disk
from skimage.restoration import denoise_nl_means, estimate_sigma

from cytomata.utils import setup_dirs, list_img_files


def preprocess_img(imgf, bkg_pc=None):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    sig = estimate_sigma(img)
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=5, patch_distance=7)
    bkg = den.copy()
    thr = threshold_li(bkg)
    cutoff = bkg_pc
    if bkg_pc is None:
        cutoff = 100 * np.mean(bkg[bkg < thr])/np.mean(bkg[bkg > thr])
    bkg[bkg >= np.percentile(bkg, cutoff)] = np.percentile(bkg, cutoff)
    bkg = gaussian(bkg, 64) + 2*sig
    bkg[bkg < 0] = 0
    img = (img - bkg)
    img[img < 0] = 0
    den = (den - bkg)
    den[den < 0] = 0
    return img, raw, bkg, den


def split_mask(maskf):
    """Subtract background and denoise fluorescence image."""
    mask = img_as_float(imread(maskf))
    mask = remove_small_objects(ndi.label(mask)[0].astype(bool), min_size=10)
    masks, n = ndi.label(mask)
    thr = masks > 0
    masks = [masks == i for i in range(1, n+1)]
    return thr, masks


def segment_object(img, segmt_local=False, factor=1, rs=None):
    """Segment out bright objects from fluorescence image."""
    if not np.any(img):
        thr = img.astype(bool)
        regs, n = None, 0
        return thr, regs, n
    if segmt_local:
        thv = threshold_local(img, block_size=3, param=3) * factor
    else:
        thv = threshold_li(img) * factor
    thr = img > thv
    thr = ndi.median_filter(thr, 2)
    thr = erosion(thr, disk(1))
    thr = ndi.binary_fill_holes(thr)
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    regs, n = ndi.label(thr)
    return thr, regs, n


def fnames_to_time(img_dir):
    img_files = list_img_files(img_dir)
    t = [float(os.path.splitext(os.path.basename(imgf))[0]) for imgf in img_files]
    return t


def calc_cmax(img_dir):
    img_files = list_img_files(img_dir)
    max_vals = [np.percentile(img_as_float(imread(imgf)), 99.999) for imgf in img_files]
    max_imgf = img_files[np.argmax(max_vals)]
    cmax = np.percentile(preprocess_img(max_imgf)[0], 99.999)
    return cmax


def process_u_csv(ty, u_csv, save_dir):
    setup_dirs(save_dir)
    t_on = []
    udf = pd.read_csv(u_csv)
    ty = np.around(ty, 0)
    tu = np.around(np.arange(ty[0], ty[-1], 1), 0)
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
    u_df = pd.DataFrame({'t': tu, 'u': u})
    u_df.to_csv(os.path.join(save_dir, 'u.csv'), index=False)
    return u_df, t_ann_img


def interp_ty(t, y):
    yf = interp1d(t, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    t = np.around(np.arange(t[0], t[-1], 1), 0)
    y = np.array([yf(ti) for ti in t])
    return t, y


def process_y_ave(y_df, save_dir):
    y_int_df = pd.DataFrame()
    for n in y_df['n'].unique():
        tn = y_df.loc[(y_df['n'] == n), 't'].values
        yn = y_df.loc[(y_df['n'] == n), 'y'].values
        tn_int, yn_int = interp_ty(tn, yn)
        n_int = np.full_like(tn_int, n)
        y_int_df_i = pd.DataFrame({'t': tn_int, 'y': yn_int, 'n': n_int})
        y_int_df = pd.concat([y_int_df, y_int_df_i], ignore_index=True)
    y_ave = []
    for ti in y_int_df['t'].unique():
        yi_ave = y_int_df.loc[(y_int_df['t'] == ti), 'y'].mean()
        y_ave.append(yi_ave)
    y_ave_df = pd.DataFrame({'t': y_int_df['t'].unique(), 'y': y_ave})
    y_ave_df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


def combine_tlapse_data(root_dir):
    combined_y = pd.DataFrame()
    n_max = 0
    data_dir = os.path.join(root_dir, 'data')
    for csvf in [fn for fn in os.listdir(data_dir) if fn.endswith('.csv')]:
        y = pd.read_csv(os.path.join(data_dir, csvf))
        y['y'] /= y['y'][0]
        y['n'] += n_max
        combined_y = pd.concat([combined_y, y])
        n_max = combined_y['n'].max() + 1
    combined_y.to_csv(os.path.join(root_dir, 'y.csv'), index=False)
