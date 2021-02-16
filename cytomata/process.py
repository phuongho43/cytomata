import os

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import img_as_float
from skimage.io import imread
from skimage.measure import label
from skimage.filters import median
from skimage.filters import (gaussian, laplace, median,
    threshold_li, threshold_yen, threshold_otsu)
from skimage.morphology import (remove_small_objects, remove_small_holes,
    disk, binary_erosion, binary_opening)
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.segmentation import clear_border

from cytomata.utils import setup_dirs


def preprocess_img(imgf):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    bkg = img.copy()
    sig = estimate_sigma(img)
    rfrac = np.percentile(raw, 25)/np.percentile(raw, 75)
    tval = threshold_li(bkg) * 1.25
    broi = bkg*(bkg < tval)
    rfrac = np.max([rfrac, 0.35])
    tval = np.percentile(broi, rfrac*100)
    bkg[bkg >= tval] = tval
    bkg = gaussian(bkg, 50)
    img = img - bkg
    img[img < 0] = 0
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=3, patch_distance=5)
    den = den - 2*sig
    den[den < 0] = 0
    return img, raw, bkg, den


def segment_object(img, factor=1, rs=None, fh=None, cb=None):
    """Segment out bright objects from fluorescence image."""
    if not np.any(img):
        return img.astype(bool)
    thv_ots = threshold_otsu(img) / 6
    thv_yen = threshold_yen(img) / 2
    thv_li = threshold_li(img) / 2
    thv = np.median([thv_ots, thv_yen, thv_li]) * factor
    thr = img > thv
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    if fh is not None:
        thr = remove_small_holes(thr.astype(bool), area_threshold=fh)
    if cb is not None:
        thr = clear_border(thr, buffer_size=cb)
    thr = median(thr, disk(3))
    return thr


def segment_clusters(img, factor=1, rs=None):
    """Segment out bright clusters from fluorescence image."""
    thv_img = threshold_li(img)
    thr_img = median(img > thv_img, disk(5))
    log = gaussian(laplace(img, mask=thr_img), sigma=2)
    samp = img[thr_img]
    samp = samp[samp < np.percentile(samp, 95)]
    thr = log > (0.1*np.std(samp) * factor)
    thr = median(thr, disk(1))
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    if not np.any(thr):
        thr = thr_img
    return thr


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