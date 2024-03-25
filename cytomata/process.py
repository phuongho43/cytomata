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
from skimage.morphology import remove_small_objects, binary_erosion, binary_opening, disk
from skimage.restoration import denoise_nl_means, estimate_sigma
import FlowCal

from cytomata.utils import setup_dirs, list_img_files


def subtract_bgd(img_fp):
    """Subract background level from fluorescence image.

    Generate an approximation of the background by thresholding away the bright features
    then applying a gaussian filter to smooth out the remainder.
    Next, subtract this background from the image.
    
    Args:
        img_fp (str): absolute filepath to image file
    Returns:
        img (2D array): processed image with background subtracted
        raw (2D array): unprocessed/original image
        bgd (2D array): approximate image of background fluorescence
    """
    img = img_as_float(imread(img_fp))
    raw = img.copy()
    bgd = img.copy()
    shift = estimate_sigma(bgd)
    thr0 = threshold_li(bgd)
    thr = threshold_li(bgd[bgd < thr0])
    fbg = bgd[bgd > thr0]
    contrast = np.round(np.std(fbg)/np.mean(fbg), 2)
    # print(img_fp, contrast)
    if contrast < 0.06:
        bgd = gaussian(bgd, 25) + 2*shift
    else:
        bgd[bgd > thr] = thr
        bgd = gaussian(bgd, 50) + shift
    bgd[bgd < 0] = 0
    img = img - bgd
    img[img < 0] = 0
    return img, raw, bgd


def split_mask(mask_fp):
    """Converts single mask of multiple ROIs into multiple masks of single ROI.

    The input mask is a binary image with 0's denoting background
    and 1's (or any number > 0) denoting multiple foreground regions of interest.
    
    Args:
        mask_fp (str): absolute filepath to the mask file
    Returns:
        regions (2D array): binary image of FALSE denoting background and TRUE denoting foreground
        masks (list of 2D arrays): list of mask images containing a single ROI in each
        centroids (dict): {n: (y, x)} centroid coordinates for each ROI (n)
    """
    mask = img_as_float(imread(mask_fp))
    mask = remove_small_objects(ndi.label(mask)[0].astype(bool), min_size=10)
    labeled, n = ndi.label(mask)
    masks = [labeled == i for i in range(1, n+1)]
    centroids = {n: ndi.center_of_mass(mi) for n, mi in enumerate(masks)}
    regions = labeled > 0
    return regions, masks, centroids


def calc_cb_max(img_dp):
    """Calculate the maximum colorbar value to use across all images in the directory.

    Useful for having a consistent colorbar scale across all images in an experiment or timelapse
    so they can be compared.

    Args:
        img_dp (str): absolute path of images directory
    Returns:
        cb_max (float): colorbar scale max value across all images
    """
    img_files = list_img_files(img_dp)
    max_vals = [np.percentile(img_as_float(imread(img_fp)), 99.9999) for img_fp in img_files]
    max_img_fp = img_files[np.argmax(max_vals)]
    cb_max = np.percentile(subtract_bgd(max_img_fp)[0], 99.9999)
    return cb_max


def norm_dF_F0(y_df):
    """Calculate the change in fluorescence relative to the resting/baseline fluorescence.

    Args:
        y_df (DataFrame): input Dataframe
            't': timepoints
            'y': fluorescence measurements
            'n': index of cell/repeat/roi/trace
    Returns:
        df (DataFrame): Dataframe with fluorescence values normalized
    """
    df = y_df.copy()
    for n in df['n'].unique():
        yn = df.loc[(df['n'] == n), 'y']
        # use timepoints before 60 as the baseline fluor.
        yn0 = df.loc[(df['n'] == n) & (df['t'] <= 60), 'y'].mean()
        df.loc[(df['n'] == n), 'y'] = (yn - yn0) / yn0
    return df


def norm_log2_fc(y_df):
    """Calculate the log2 fold change relative to the resting/baseline fluorescence.

    Args:
        y_df (DataFrame): input Dataframe
            't': timepoints
            'y': fluorescence measurements
            'n': index of cell/repeat/roi/trace
    Returns:
        df (DataFrame): Dataframe with fluorescence values normalized
    """
    df = y_df.copy()
    for n in df['n'].unique():
        yn = df.loc[(df['n'] == n), 'y']
        # use timepoints before 60 as the baseline fluor.
        yn0 = df.loc[(df['n'] == n) & (df['t'] <= 60), 'y'].mean()
        df.loc[(df['n'] == n), 'y'] = np.log2(yn/yn0)
    return df


def process_u_csv(tt, u_csv_fp):
    """Process an input signal csv file into timepoints format.

    The input signal csv file has two columns (ta, tb).
    ta corresponds to timepoints where the input signal is turned on.
    tb corresponds to timepoints where the input signal was turned off.
    For the timepoints in tt, generate an array of same length uu
    with 0's for when the signal is off and 1's for when the signal is on.

    Args:
        tt (1D array): timepoints (spanning the same timerange as the fluorescence images)
        u_csv_fp (str): absolute filepath of the input signals csv file
    Returns:
        u_df (DataFrame): timepoints and input signals data
    """
    u_df = pd.read_csv(u_csv_fp)
    u_ta = np.round(u_df['ta'].values, 1)
    u_tb = np.round(u_df['tb'].values, 1)
    tt = np.round(tt, 1)
    uu = np.zeros_like(tt)
    for ta, tb in zip(u_ta, u_tb):
        if ta > tt[-1]:
            continue
        ia = np.where(tt == ta)[0][0]
        ib = np.where(tt == tb)[0][0]
        uu[ia:ib+1] = 1.0
    u_df = pd.DataFrame({'t': tt, 'u': uu})
    return u_df


def load_fc_data(fcs_fp, channel):
    """Load flow cytometry measurements for a given fluorescence channel from an FCS file.

    Castillo-Hair S.M., Sexton J.T., et al. FlowCal: A User-Friendly, Open Source Software Tool
    for Automatically Converting Flow Cytometry Data from Arbitrary to Calibrated Units.
    ACS Synth. Biol. 2016.

    Args:
        fcs_fp (str): absolute filepath for the FCS file
        channel (str): fluorescence channel name e.g. 'FL4-A'
    Returns:
        y (1D array): fluorescence readings (a.u.)
    """
    y = FlowCal.io.FCSData(fcs_fp)
    # print(y.channels)
    y = FlowCal.transform.to_rfi(y)
    y = y[:, channel]
    return y


def values_at_timepoints(y_dfs, timepoints):
    """Fetch response values from multiple expt groups at specific timepoints for comparison.

    Args:
        y_dfs (list of DataFrames): list of dataframes corresponding to various expt groups
        timepoints (list of floats): list of the timepoints of interest
    Returns:
        tp_df (DataFrame):
            't': timepoints
            'y': response values
            'h': index of the expt groups
    """
    tp_df = pd.DataFrame()
    for h, y_df in enumerate(y_dfs):
        for tp in timepoints:
            tp_df_i = y_df.loc[(y_df['t'] == tp), ['y']]
            tp_df_i['h'] = np.ones_like(tp_df_i['y']) * h
            tp_df_i['t'] = np.ones_like(tp_df_i['y']) * tp
            tp_df = pd.concat([tp_df, tp_df_i], ignore_index=True)
    return tp_df


def approx_half_life(t, y):
    """Approximate the half life of a decaying spike response.

    Half life is the time it takes for the response to return halfway back to baseline
    from maximal excitation.

    Args:
        t (1D array): timepoints
        y (1D array): response values
    Returns:
        t_half (float): half life
    """
    i_exc = np.argmax(np.abs(y[:18]))
    t_exc = t[i_exc]
    y_exc = y[i_exc]
    y_bas = np.mean(y[:i_exc-5])
    tp = t[i_exc:]
    tp = tp - np.min(tp)
    yp = y[i_exc:]
    yf = interp1d(tp, yp, 'linear')
    ti = np.arange(tp[0], tp[-1], 0.1)
    yi = yf(ti)
    y_half = (y_bas + y_exc)/2
    idx = np.argmin(np.abs((yi - y_half)))
    t_half = np.round(ti[idx], 1)
    return t_half


def calc_t_half_data(y_dfs):
    """Calculate the half life for multiple expt groups.

    Args:
        y_dfs (list of DataFrames): list of dataframes corresponding to various expt groups
    Returns:
        t_half_df (DataFrame):
            'x': index of expt groups
            'y': half life values
    """
    t_half_vals = []
    h_vals = []
    for h, y_df_h in enumerate(y_dfs):
        for n in y_df_h['n'].unique():
            try:
                y_df_hn = y_df_h.loc[(y_df_h['n'] == n)]
                t_vals = y_df_hn['t'].values
                y_vals = y_df_hn['y'].values
                t_half_val = approx_half_life(t_vals, y_vals)
                t_half_vals.append(t_half_val)
                h_vals.append(h)
            except Exception:  # sometimes the data is too messy to calculate
                pass
    t_half_df = pd.DataFrame({'x': h_vals, 'y': t_half_vals})
    return t_half_df
