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
from skimage.filters import (gaussian, median, threshold_li, threshold_local, threshold_otsu)
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
        fig, ax = plt.subplots(figsize=(18,12))
        axim = ax.imshow(img, cmap='gray')
        axim.set_clim(0.0, cmax)
        if t_unit:
            t_text = 't = ' + fname + t_unit
            ax.text(0.05, 0.95, t_text, ha='left', va='center', fontsize=20, transform=ax.transAxes)
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
        ax.grid(False)
        ax.axis('off')
        cb = fig.colorbar(axim, pad=0.01, format='%.3f', extend='both', extendrect=True,
                ticks=np.linspace(np.min(img), cmax, 10))
        cb.outline.set_linewidth(0)
        fig.tight_layout(pad=0)
        if thr is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(thr, linewidths=0.1, colors='r')
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, fname + '.png'),
            dpi=300, bbox_inches='tight', pad_inches=0)
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
        thv = threshold_local(img, block_size=3, param=1, offset=-3) * factor
    else:
        thv = threshold_li(img) * factor
    thr = img > thv
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    reg, n = ndi.label(thr)
    return thr, reg, n


def count_cells(img_dir, save_dir, segmt_factor=1, remove_small=None, fill_holes=None):
    """Count number of cells in fluorescent image."""
    data = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        img, raw, bkg, den = preprocess_img(imgf)
        cmax_i = np.percentile(img, 100)
        # plot_bkg_profile(fname, save_dir, raw, bkg)
        thr, reg, n = segment_object(den, segmt_local=True, factor=segmt_factor, rs=remove_small)
        data.append({'file': fname, 'n_cells': n})
        img_save_dir = os.path.join(save_dir, 'imgs')
        cell_den = plot_cell_img(den, thr, fname, img_save_dir, cmax=cmax_i)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


if __name__ == '__main__':
    ## Set Parameters #########################################
    img_dir = '/home/phuong/data/count/20220504/'
    segmt_factor = 1  # [float] Tune the thresholding. Higher => exclude dimmer regions | Lower => include dimmer regions
    remove_small = None  # [int] Excludes regions smaller than the specified area in pixels squared
    ###########################################################

    root_dir = Path(img_dir).parent.absolute()
    img_dir = Path(img_dir).name
    save_dir = os.path.join(root_dir, img_folder + '-results')
    count_cells(img_dir, save_dir, segmt_factor=segmt_factor, remove_small=remove_small, fill_holes=None)