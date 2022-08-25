import os
import warnings

import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage import img_as_ubyte

from cytomata.utils import setup_dirs, custom_styles, custom_palette


def plot_cell_img(img, thr, fname, save_dir, cmax, sig_ann=False, t_unit=None, sb_microns=None):
    setup_dirs(save_dir)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(18,12))
        axim = ax.imshow(img, cmap='turbo')
        axim.set_clim(0.0, cmax)
        if t_unit:
            t_text = 't = ' + fname + t_unit
            ax.text(0.05, 0.95, t_text, ha='left', va='center', color='white', fontsize=20, transform=ax.transAxes)
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
                ax.contour(thr, linewidths=0.1, colors='w')
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, fname + '.png'),
            dpi=100, bbox_inches='tight', pad_inches=0)
        cell_img = img_as_ubyte(np.array(fig.canvas.renderer._renderer))
        plt.clf()
        plt.close('all')
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
        plt.clf()
        plt.close('all')


def plot_uy(y_df, u_df, save_dir, t_unit='s', ulabel='stim.'):
    setup_dirs(save_dir)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        if u_df is not None:
            fig, (ax0, ax) = plt.subplots(
                2, 1, sharex=True, figsize=(16, 10),
                gridspec_kw={'height_ratios': [1, 8]}
            )
            ax0.plot(u_df['t'], u_df['u'])
            ax0.set_yticks([0, 1])
            ax0.set_ylabel(ulabel)
        else:
            fig, ax = plt.subplots(figsize=(16,8))
        sns.lineplot(data=y_df, x="t", y="y", color='#785EF0')
        ax.text(0.96, 1.01, 'n={}'.format(y_df['n'].nunique()), ha='left', va='center', fontsize=18, transform=ax.transAxes)
        ax.set_xlabel('Time ({})'.format(t_unit))
        ax.set_ylabel('Fold Change')
        ax.locator_params(axis='x', nbins=10)
        ax.locator_params(axis='y', nbins=8)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'y.png'), dpi=300, bbox_inches='tight', transparent=False)
        plt.clf()
        plt.close('all')
