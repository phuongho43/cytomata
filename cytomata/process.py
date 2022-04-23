import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.filters import gaussian, threshold_li, threshold_local
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt

from cytomata.utils import setup_dirs
import time


def preprocess_img(imgf):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    sig = estimate_sigma(img)
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=5, patch_distance=7)
    bkg = den.copy()
    thr = threshold_li(bkg)
    chop = 100 * np.mean(bkg[bkg < thr])/np.mean(bkg[bkg > thr])
    bkg[bkg >= np.percentile(bkg, chop)] = np.percentile(bkg, chop)
    bkg = gaussian(bkg, 64) + sig
    bkg[bkg < 0] = 0
    img = (img - bkg) / bkg
    img[img < 0] = 0
    den = (den - bkg) / bkg
    den[den < 0] = 0
    return img, raw, bkg, den


def segment_object(img, segmt_local=False, factor=1, rs=None):
    """Segment out bright objects from fluorescence image."""
    if not np.any(img):
        thr = img.astype(bool)
        reg, n = None, 0
        return thr, reg, n
    if segmt_local:
        thv = threshold_local(img, block_size=5, param=24) * factor
    else:
        thv = threshold_li(img) * factor
    thr = img > thv
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    reg, n = ndi.label(thr)
    return thr, reg, n


def process_u_csv(ty, u_csv, save_dir):
    setup_dirs(save_dir)
    t_on = []
    udf = pd.read_csv(u_csv)
    ty = np.around(ty, 1)
    tu = np.around(np.arange(ty[0], ty[-1], 0.1), 1)
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(tu)
    for ta, tb in zip(uta, utb):
        if ta > ty[-1]:
            continue
        t_on += list(np.arange(round(ta, 1), round(tb, 1) + 0.01, 0.1))
        ia = list(tu).index(ta)
        ib = list(tu).index(tb)
        u[ia:ib+1] = 1
    t_ann_img = []
    for tbl in t_on:
        t_ann_img.append(min(ty, key=lambda ti : abs(ti - tbl)))
    t_ann_img = np.unique(t_ann_img)
    u_data = np.column_stack((tu, u))
    u_path = os.path.join(save_dir, 'u.csv')
    np.savetxt(u_path, u_data, delimiter=',', header='t,u', comments='')
    return tu, u, t_ann_img