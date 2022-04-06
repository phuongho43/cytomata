import os
import time
import imghdr
import warnings
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from natsort import natsorted, ns
from scipy import ndimage as ndi

from matplotlib import font_manager
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from skimage import img_as_float, img_as_ubyte, img_as_uint
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from skimage.filters import (gaussian, median, threshold_li, threshold_local)
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma


custom_palette = ['#648FFF', '#FE6100', '#DC267F', '#785EF0', '#FFB000']

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
    if (np.percentile(bkg, 99) - np.percentile(bkg, 1))/np.median(bkg) < 1:
        broi = broi[(broi > np.percentile(broi, 99))]
    else:
        broi = broi[(broi > np.percentile(broi, 50))]
    tval = threshold_li(broi)
    bkg[bkg >= tval] = tval
    bkg = gaussian(bkg, 64) + sig
    bkg[bkg < 0] = 0
    img = (img - bkg) / bkg
    img[img < 0] = 0
    den = (den - bkg) / bkg
    den[den < 0] = 0
    return img, raw, bkg, den


def segment_object(img, local=False, factor=1, rs=None):
    """Segment out bright objects from fluorescence image."""
    if not np.any(img):
        thr = img.astype(bool)
        reg, n = None, 0
        return thr, reg, n
    if local:
        thv = threshold_local(img, block_size=5, param=24) * factor
    else:
        thv = threshold_li(img) * factor
    thr = img > thv
    if rs is not None:
        thr = remove_small_objects(ndi.label(thr)[0].astype(bool), min_size=rs)
    reg, n = ndi.label(thr)
    return thr, reg, n


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
            thr, reg, n = segment_object(den, local=segmt_local, factor=segmt_factor, rs=remove_small)
            img_save_dir = os.path.join(save_dir, 'outlined')
            cell_den = plot_cell_img(den, thr, fname, img_save_dir, cmax=cmax_i, sb_microns=sb_microns)
            mi = np.mean(img[thr])
            if np.isnan(mi):
                mi = np.mean(img)
        data = {'fname': fname, 'mean_int': mi, 'num_cells': n}
        return data
    setup_dirs(os.path.join(save_dir, 'subtracted'))
    data = []
    # for i, imgf in enumerate(list_img_files(img_dir)):
    #     data = img_task(data, i, imgf)
    #     print(i)
    data = Parallel(n_jobs=os.cpu_count())(delayed(img_task)(data, i, imgf) for i, imgf in enumerate(list_img_files(img_dir)))
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


def combine_groups(root_dir, ch):
    df = pd.DataFrame(columns=['Group', 'Response'])
    for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(root_dir)][0], alg=ns.IGNORECASE)):
        print(i, data_dir)
        y_csv = os.path.join(root_dir, data_dir, ch + '-results', 'y.csv')
        y_data = pd.read_csv(y_csv)
        rs = y_data['mean_int'].values
        gr = np.full_like(rs, i+1)
        di = pd.DataFrame(np.column_stack([gr, rs]), columns=['Group', 'Response'])
        df = pd.concat([df, di], ignore_index=True)
        # df = df.append(di, ignore_index=True)
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


def combine_before_after(root_dir):
    df = pd.DataFrame(columns=['Group', 'Timepoint', 'Response'])
    for tpoint in ['before', 'after']:
        tp_dir = os.path.join(root_dir, tpoint)
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(tp_dir)][0])):
            y_csv = os.path.join(tp_dir, data_dir, 'results', 'y.csv')
            y_data = pd.read_csv(y_csv)
            rs = y_data['mean_int'].values
            gr = np.full_like(rs, i+1)
            tp = np.full_like(rs, tpoint)
            di = pd.DataFrame(np.column_stack([gr, tp, rs]), columns=['Group', 'Timepoint', 'Response'])
            df = pd.concat([df, di], ignore_index=True)
    df.to_csv(os.path.join(root_dir, 'y.csv'), index=False)


def plot_before_after(root_dir, group_labels, group_order, figsize=(24, 8)):
    df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    df = df[df.Group.isin(group_order)]
    df_fc = df.copy()
    fc_vals = {}
    for gr in df.Group.unique():
        mean_before = df.loc[(df['Group'] == gr) & (df['Timepoint'] == 'before'), 'Response'].mean()
        mean_after = df.loc[(df['Group'] == gr) & (df['Timepoint'] == 'after'), 'Response'].mean()
        fc_vals[gr] = str(round(mean_after/mean_before, 2)) + r'$\times$'
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
        ax.legend(handles, ['before', 'after'], loc='best', prop={"size": 20}, frameon=True, shadow=False)
        fig_name = 'y_' + '-'.join([str(int(g)) for g in df.Group.unique()]) + '.png'
        plt.savefig(os.path.join(root_dir, fig_name), dpi=200, transparent=False, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    ## Set Parameters ##
    root_dir = '/home/phuong/data/FPs/Lenti/20220403/'
    img_folder = 'mCherry' # (e.g. Default, DIC, GFP, etc. Make sure they're named the same for between group folders)
    group_labels = [  # x-axis label for each group. '\n' = newline
        'HEK293T',
        'L929',
    ]
    group_order = [1, 2] # Order to plot each group on the bargraph
    sb_microns = 110  # [float] Specify scalebar label in microns or None for no scalebar
    cmax = None  # [float] Colorbar upper limit value. Leave None to auto calculate.
    segmt = True  # [bool] Whether to perform object segmentation and calculate mean intensity of only pixels in those regions or don't and calculate it using every pixel in the whole image
    segmt_local = True  # [bool] Whether to use local thresholding or global thresholding for the segmentation
    segmt_factor = 1  # [float] Tune the thresholding. Higher => exclude dimmer regions | Lower => include dimmer regions
    remove_small = 100  # [int] Excludes regions smaller than the specified area in pixels squared
    before_after = False  # [bool] Set True if "before-after" images for each group were taken.
    
    ## General File Structure ##
    #---Group_Dir
    #-----Imgs_Dir/Channel_Dir
    #-------1.tif (can be named whatever)
    #-------2.tif
    #-------...etc...
    #-----Results_Dir (will be generated by this script)
    #-------debug (for checking background subtraction; blue = img | orange = bkg)
    #-------subtracted (tif images with background subtracted)
    #-------denoised (bkg subtracted + denoising algorithm applied)
    #-------outlined (denoised image with object segmentation regions outlined in white)
    #-------y.csv (data extracted from image)

    #### Comparison of Groups ##
    ## File Structure if before_after = False ##
    #---Root_Dir
    #-----group1 # (can be named whatever)
    #-------channel1 # (e.g. Default, DIC, GFP, etc. Make sure they're named the same for between group folders)
    #---------1.tif (can be named whatever)
    #---------2.tif
    #---------...etc...
    #-------channel2
    #---------1.tif (can be named whatever)
    #---------2.tif
    #---------...etc...
    #-----group2 # can be named whatever
    #-------channel1
    #---------1.tif
    #---------2.tif
    #---------...etc...
    #-------channel2
    #---------1.tif
    #---------2.tif
    #---------...etc...
    for group_dir in natsorted(os.listdir(root_dir)):
        if group_dir == ".directory":
            continue
        print(group_dir)
        img_dir = os.path.join(root_dir, group_dir, img_folder)
        save_dir = os.path.join(root_dir, group_dir, img_folder + '-results')
        process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax,
            segmt=segmt, segmt_local=segmt_local, segmt_factor=segmt_factor, remove_small=remove_small)

    combine_groups(root_dir, img_folder)
    plot_groups(root_dir, group_labels, group_order=group_order, figsize=(len(group_labels)*6, 8))


    #### Before-After Fold Change ##
    ## File Structure if before_after = True ##
    #---Root_Dir
    #-----before # has to be a folder named "before"
    #-------group1 # (can be named whatever. Make sure they're named the same between before/after folders)
    #---------1.tif (can be named whatever)
    #---------2.tif
    #---------...etc...
    #-------group2
    #---------1.tif
    #---------2.tif
    #---------...etc...
    #-----after # has to be a folder named "after"
    #-------group1
    #---------1.tif
    #---------2.tif
    #---------...etc...
    #-------group2
    #---------1.tif
    #---------2.tif
    #---------...etc...
    # for t in ['before', 'after']:
    #     tp_dir = os.path.join(root_dir, t)
    #     for group_dir in natsorted(os.listdir(tp_dir)):
    #         if group_dir == ".directory":
    #             continue
    #         print(group_dir)
    #         img_dir = os.path.join(tp_dir, group_dir, img_folder)
    #         save_dir = os.path.join(tp_dir, group_dir, 'results')
    #         process_fluo_images(img_dir, save_dir, sb_microns=sb_microns, cmax=cmax,
    #             segmt=segmt, segmt_local=segmt_local, segmt_factor=segmt_factor, remove_small=remove_small)
    # combine_before_after(root_dir)
    # plot_before_after(root_dir, group_labels, group_order=group_order, figsize=(len(group_labels)*6, 8))


