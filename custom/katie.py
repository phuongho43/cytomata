import os
import time
import imghdr
import warnings
import itertools
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from natsort import natsorted, ns
from scipy import ndimage as ndi

from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skimage import img_as_float, img_as_ubyte, img_as_uint
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from skimage.filters import (gaussian, median, threshold_li, threshold_local)
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.measure import regionprops


custom_palette = ['#648FFF', '#DC267F', '#FE6100', '#FFB000', '#785EF0']

custom_styles = {
    'figure.figsize': (16, 8),
    'text.color': '#212121',
    'axes.titleweight': 'bold',
    'axes.titlesize': 32,
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 28,
    'axes.labelpad': 10,
    'axes.labelcolor': '#212121',
    'axes.labelweight': 600,
    'axes.linewidth': 3,
    'axes.edgecolor': '#212121',
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'lines.linewidth': 5
}


def list_img_files(dir):
    return [os.path.join(dir, fn) for fn in natsorted(os.listdir(dir), key=lambda y: y.lower())
        if imghdr.what(os.path.join(dir, fn)) in ['tiff', 'jpeg', 'png', 'gif']]


def setup_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def plot_cell_img(img, thr, fname, save_dir, cmax, sig_ann=False, t_unit=None, sb_microns=None):
    setup_dirs(save_dir)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(10,8))
        axim = ax.imshow(img, cmap='turbo')
        axim.set_clim(0.0, cmax)
        if t_unit:
            t_text = 't = ' + fname + t_unit
            ax.annotate(t_text, (16, 32), color='white', fontsize=20)
        if sb_microns is not None:
            fontprops = font_manager.FontProperties(size=20)
            asb = AnchoredSizeBar(ax.transData, 100, u'{}\u03bcm'.format(sb_microns),
                color='white', size_vertical=2, fontproperties=fontprops,
                loc='lower left', pad=0.1, borderpad=0.5, sep=5, frameon=False)
            ax.add_artist(asb)
        if sig_ann:
            w, h = img.shape
            ax.add_patch(Rectangle((3, 3), w-7, h-7,
                linewidth=5, edgecolor='#648FFF', facecolor='none'))
        # ax.grid(False)
        plt.rcParams['axes.grid'] = False
        ax.axis('off')
        cb = fig.colorbar(axim, pad=0.01, format='%.3f',
            extend='both', extendrect=True, extendfrac=0.03)
        cb.outline.set_linewidth(0)
        fig.tight_layout(pad=0)
        if thr is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(thr, linewidths=0.1, colors='w')
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, fname + '.png'),
            dpi=100, bbox_inches='tight', pad_inches=0)
        cell_img = img_as_ubyte(np.array(fig.canvas.renderer._renderer))
        plt.close(fig)
        return cell_img


def plot_bkg_profile(fname, save_dir, img, bkg):
    setup_dirs(os.path.join(save_dir, 'debug'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
        row_i = np.random.choice(bg_rows.shape[0])
        bg_row = bg_rows[row_i]
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(img[bg_row, :])
        ax.plot(bkg[bg_row, :])
        ax.set_title(str(bg_row))
        bg_path = os.path.join(save_dir, 'debug', '{}.png'.format(fname))
        fig.savefig(bg_path, bbox_inches='tight', transparent=False, dpi=100)
        plt.close(fig)


def preprocess_img(imgf):
    """Subtract background and denoise fluorescence image."""
    img = img_as_float(imread(imgf))
    raw = img.copy()
    sig = estimate_sigma(img)
    den = denoise_nl_means(img, h=sig, sigma=sig, patch_size=5, patch_distance=7)
    bkg = den.copy()
    thr = threshold_local(bkg, block_size=5, param=24)
    broi = bkg * (bkg < thr)
    if (np.percentile(bkg, 99.9) - np.percentile(bkg, 0.1))/np.percentile(bkg, 0.1) < 0.5:
        broi = broi[(broi > np.percentile(broi, 99))]
    else:
        broi = broi[(broi > np.percentile(broi, 40))]
    tval = threshold_li(broi)
    bkg[bkg >= tval] = tval
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


def process_fluo_images(img_dir, save_dir, sb_microns=11,
    cmax=None, segmt_local=False, segmt_factor=1, remove_small=None):
    """Analyze fluorescence 10x images and generate figures."""
    def img_task(data, imgf):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        thr, reg, n = segment_object(den, segmt_local=segmt_local, factor=segmt_factor, rs=remove_small)
        cmax_i = cmax
        if cmax is None:
            cmax_i = np.percentile(img, 99.99)
        plot_bkg_profile(fname, save_dir, raw, bkg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(save_dir, 'subtracted', fname + '.tiff'), img_as_uint(rescale_intensity(img)))
        plot_cell_img(den, None, fname, os.path.join(save_dir, 'denoised'), cmax=cmax_i, sb_microns=sb_microns)
        plot_cell_img(den, thr, fname, os.path.join(save_dir, 'outlined'), cmax=cmax_i, sb_microns=sb_microns)
        y = np.mean(img[thr])
        if np.isnan(y):
            np.mean(img)
        data = {'i': fname, 'y': y}
        return data
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    data = []
    data = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(data, imgf) for imgf in list_img_files(img_dir))
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


if __name__ == '__main__':
    ## File Structure ##
    #--Root_Dir
    #----Imgs_Dir
    #------1.tif
    #------2.tif
    #------...etc...
    #----Results_Dir (will be generated by this script)
    #------debug (for checking background subtraction; blue = img | orange = bkg; the orange line should look like the "base" of the blue hills)
    #------subtracted (tif images with background subtracted)
    #------denoised (bkg subtracted + denoising algorithm applied)
    #------outlined (denoised image with segmented regions outlined in white)
    #------y.csv (mean intensity of pixels in the segmented regions)
    
    ## Set Parameters #########################################
    img_dir = '/home/phuong/data/katie/20220420/mCherry/'
    sb_microns = 11  # [float] Specify scalebar label in microns or None for no scalebar
    cmax = None  # [float] Colorbar upper limit value. Leave None to auto calculate.
    segmt_factor = 1  # [float] Tune the thresholding. Higher => exclude dimmer regions | Lower => include dimmer regions
    remove_small = 2500  # [int] Excludes regions smaller than the specified area in pixels squared
    ###########################################################

    root_dir = Path(img_dir).parent.absolute()
    img_folder = Path(img_dir).name
    save_dir = os.path.join(root_dir, img_folder + '-results')
    process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax,
        segmt_factor=segmt_factor, remove_small=remove_small)