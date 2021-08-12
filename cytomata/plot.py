import os
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage import img_as_ubyte

from cytomata.utils import setup_dirs, custom_styles, custom_palette


def plot_cell_img(img, thr, fname, save_dir, cmax, sig_ann=False, t_unit=None, sb_microns=None):
    setup_dirs(os.path.join(save_dir, 'imgs'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(10,8))
        axim = ax.imshow(img, cmap='inferno')
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
                linewidth=5, edgecolor='#2196F3', facecolor='none'))
        ax.grid(False)
        ax.axis('off')
        cb = fig.colorbar(axim, pad=0.01, format='%.3f',
            extend='both', extendrect=True, extendfrac=0.03)
        cb.outline.set_linewidth(0)
        fig.tight_layout(pad=0)
        if thr is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(thr, linewidths=0.3, colors='w')
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'imgs', fname + '.png'),
            dpi=100, bbox_inches='tight', pad_inches=0)
        cell_img = img_as_ubyte(np.array(fig.canvas.renderer._renderer))
        plt.close(fig)
        return cell_img


def plot_bkg_profile(fname, save_dir, img, bkg):
    setup_dirs(os.path.join(save_dir, 'bg_sub'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
        row_i = np.random.choice(bg_rows.shape[0])
        bg_row = bg_rows[row_i]
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(img[bg_row, :])
        ax.plot(bkg[bg_row, :])
        ax.set_title(str(bg_row))
        bg_path = os.path.join(save_dir, 'bg_sub', '{}.png'.format(fname))
        fig.savefig(bg_path, bbox_inches='tight', transparent=False, dpi=100)
        plt.close(fig)


def plot_uy(t, y, tu, u, save_dir, t_unit='s', ulabel='BL'):
    setup_dirs(save_dir)
    t = np.array(t)
    y = np.array(y)
    tu = np.array(tu)
    u = np.array(u)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        if len(tu) > 0:
            fig, (ax0, ax) = plt.subplots(
                2, 1, sharex=True, figsize=(16, 10),
                gridspec_kw={'height_ratios': [1, 8]}
            )
            ax0.plot(tu, u)
            ax0.set_yticks([0, 1])
            ax0.set_ylabel(ulabel)
        else:
            fig, ax = plt.subplots(figsize=(16,8))
        ax.plot(t, y, color='#d32f2f')
        ax.set_xlabel('Time ({})'.format(t_unit))
        ax.set_ylabel('Ave Fl. Intensity')
        ytiks, ystep = np.linspace(np.min(y), np.max(y), 6, endpoint=True, retstep=True)
        ylim = (ytiks[0] - ystep/4, ytiks[-1] + ystep/4)
        ax.set_yticks(ytiks)
        ax.set_ylim(ylim)
        ax1 = ax.twinx()
        ax1.plot(t, y/np.mean(y[:3]), color='#d32f2f')
        ax1.set_yticks(ytiks/np.mean(y[:3]))
        ax1.set_ylim(ylim/np.mean(y[:3]))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax1.set_ylabel('Fold Change')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'y.png'),
            dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)