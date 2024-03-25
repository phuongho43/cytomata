import os
import warnings
import ast

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib import font_manager
from matplotlib import path as mpath
from matplotlib import patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from natsort import natsorted
from joblib import Parallel, delayed

from cytomata.utils import setup_dirs, custom_styles, custom_palette


def plot_cell_img(img, fig_fp,
    cb_max=None, t_unit=None, sb_microns=None, sig_ann=False,
    regions=None, centroids=None, roi_n0=0):
    """Generate colored and annotated fluorescence microscopy images.

    Args:
        img (2D array): processed fluorescence image
        fig_fp (str): absolute filepath for saving the figure image
        cb_max (float): maximum value for the colormap scale and colorbar
        t_unit (str): unit for the annotated timestamp
        sb_microns (int): the length in microns equivalent to 200 pixels
            for the scalebar text annotation (specify None for no scalebar)
        sig_ann (bool): whether to draw a thing blue outline around the whole image to denote
            input signal/stimuli exposure
        regions (2D array): binary image for drawing white outlines denoting regions of interest
            TRUE = foreground; FALSE = background
        centroids (dict): {n: (y, x)} coordinates of centroids for annotating ROIs
            with their assigned number
        roi_n0 (int): starting number or the roi counter
    """
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots(figsize=(24, 16))
        axim = ax.imshow(img, cmap='turbo')
        axim.set_clim(0.0, cb_max)
        if t_unit is not None:
            timepoint = os.path.splitext(os.path.basename(fig_fp))[0]
            t_text = 't = ' + timepoint + t_unit
            ax.text(0.05, 0.95, t_text, ha='left', va='center_baseline',
                color='white', fontsize=64, weight='bold', transform=ax.transAxes)
        if sb_microns is not None:
            fontprops = font_manager.FontProperties(size=80, weight='bold')
            asb = AnchoredSizeBar(ax.transData, 200, u'{}\u03bcm'.format(sb_microns),
                color='white', size_vertical=8, fontproperties=fontprops,
                loc='lower left', pad=0, borderpad=0.2, sep=10, frameon=False)
            ax.add_artist(asb)
        if sig_ann:
            w, h = img.shape
            ax.add_patch(mpatches.Rectangle((2, 2), w-7, h-7,
                linewidth=10, edgecolor='#648FFF', facecolor='none'))
        ax.grid(False)
        ax.axis('off')
        cb = fig.colorbar(axim, pad=0.005, format='%.3f',
            extend='both', extendrect=True, ticks=np.linspace(np.min(img), cb_max, 10))
        cb.outline.set_linewidth(0)
        cb.ax.tick_params(labelsize=84)
        fig.tight_layout(pad=0)
        if regions is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(regions, linewidths=3, colors='w')
            if centroids is not None:
                for num, (y, x) in centroids.items():
                    ax.annotate(str(roi_n0+num), xy=(x, y), xycoords='data',
                        color='white', fontsize=48, ha='center', va='center_baseline')
        fig.canvas.draw()
        fig.savefig(fig_fp, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')


def plot_bgd_prof(img, bgd, fig_fp):
    """Plot the pixel value profile of the raw image and approximated background.

    Done to verify the quality of the background subtraction for fluorescence image processing.

    img (2D array): the raw/unprocessed image before background subtraction
    bgd (2D array): the approx background image
    fig_fp (str): absolute filepath for saving this figure
    """
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
        row_i = np.random.choice(bg_rows.shape[0])
        bg_row = bg_rows[row_i]
        fig, ax = plt.subplots(figsize=(24, 20))
        ax.plot(img[bg_row, :], color='#648FFF')
        ax.plot(bgd[bg_row, :], color='#785EF0')
        fig.savefig(fig_fp, pad_inches=0, bbox_inches='tight', transparent=False, dpi=100)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')


def plot_uy(fig_fp, y_df, u_df=None, z_df=None, x='t', u='u', y='y', h='h',
        xlabel='Time (s)', ylabel=r'$\mathbf{\Delta F/F_{0}}$', ulabel='BL',
        group_labels=None, annotation=None, ymin=None, ymax=None,
        palette=custom_palette, lgd_loc='best', dpi=100):
    """
    """
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        if u_df is not None:
            fig, (ax0, ax) = plt.subplots(
                2, 1, sharex=True, figsize=(24, 16),
                gridspec_kw={'height_ratios': [1, 8]}
            )
            sns.lineplot(data=u_df, x=x, y=u, ax=ax0, errorbar=None, color='#648FFF')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel(ulabel)
        else:
            fig, ax = plt.subplots(figsize=(24, 14))
        if 'h' in y_df.columns:
            sns.lineplot(data=y_df, x=x, y=y, ax=ax, hue=h, style=h,
                estimator='mean', errorbar=('se', 1.96), palette=palette)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, group_labels, loc=lgd_loc,
                markerscale=4, frameon=True, shadow=False, handletextpad=0.2,
                borderpad=0.2, labelspacing=0.2, handlelength=1)
        else:
            if z_df is not None:
                sns.lineplot(data=y_df, x=x, y=y, ax=ax, color=palette[0], zorder=1)
                sns.scatterplot(data=z_df, x=x, y=y, ax=ax, color=palette[1], s=400, linewidth=0)
                handles = [
                    mpl.lines.Line2D([], [], color=palette[1],
                        marker='o', markersize=8, linewidth=0),
                    mpl.lines.Line2D([], [], color=palette[0], linewidth=16),
                ]
                ax.legend(handles, group_labels, loc=lgd_loc,
                    markerscale=4, frameon=True, shadow=False, handletextpad=0.2,
                    borderpad=0.2, labelspacing=0.2, handlelength=1)
            else:
                sns.lineplot(data=y_df, x=x, y=y, ax=ax,
                    estimator='mean', errorbar=('se', 1.96), color=palette[0],
                    marker='o', markeredgecolor='#212121', markersize=12, markeredgewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis='x', nbins=10)
        ax.locator_params(axis='y', nbins=10)
        ymin_i, ymax_i = ax.get_ylim()
        if ymax is None:
            ymax = ymax_i
        if ymin is None:
            ymin = ymin_i
        ax.set_ylim(ymin, ymax)
        if annotation is not None:
            ax.text(annotation['x'], annotation['y'], annotation['text'],
                ha='left', va='center_baseline', color=annotation['color'],
                fontsize=annotation['size'], transform=ax.transAxes)
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(fig_fp, pad_inches=0.3, dpi=dpi, bbox_inches='tight', transparent=False)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')


def plot_groups(y_df, fig_fp, group_labels, class_labels=None, x='group', y='response', h='class',
        ylabel='', xlabel='', ymin=None, ymax=None, palette=custom_palette,
        lgd_loc='best', dpi=100, figsize=(24, 16), rc_custom=None):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        if rc_custom is not None:
            sns.set_context(rc=rc_custom)
        fig, ax = plt.subplots(figsize=figsize)
        dodge_0 = False
        dodge_1 = False
        if class_labels:
            dodge_0 = True
            dodge_1 = 0.4
        glen = len(group_labels)
        clen = 2 if class_labels is None else len(class_labels)
        capsize = 0.015*(glen+1)*(clen+1)
        sns.stripplot(x=x, y=y, hue=h, data=y_df, ax=ax,
            palette=palette, dodge=dodge_0, jitter=0.1, size=42, linewidth=2,
            edgecolor='#212121', alpha=0.8, zorder=1)
        sns.pointplot(x=x, y=y, hue=h, data=y_df, ax=ax,
            estimator='mean', errorbar=('se', 1.96),
            color='#212121', dodge=dodge_1, markers='.', errwidth=10,
            join=False, capsize=capsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(group_labels)
        # ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        # ymin, ymax = y_df[y].min(), y_df[y].max()
        # tick_range = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        # ax.yaxis.set_ticks(tick_range)
        # tick_range = np.arange(np.floor(ymin), np.ceil(ymax))
        # ax.yaxis.set_ticks(
        #     [np.log10(x) for p in tick_range
        #     for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        # ax.set_yscale('log')
        ymin_i, ymax_i = ax.get_ylim()
        if ymax is None:
            ymax = ymax_i
        if ymin is None:
            ymin = ymin_i
        ax.set_ylim(ymin, ymax)
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        ax.locator_params(axis='y', nbins=8)
        if class_labels is not None:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, class_labels, loc=lgd_loc,
                markerscale=4, frameon=True, shadow=False, handletextpad=0.1, borderpad=0.1,
                labelspacing=0.2, handlelength=1)
        else:
            ax.get_legend().remove()
        plt.savefig(fig_fp, pad_inches=0.2, dpi=dpi, transparent=False, bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')


def plot_fc_hist(y_df, gateline, save_path, group_labels, xlabel, palette=custom_palette, figsize=(32, 16)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0),
                'xtick.major.size': 12, 'xtick.minor.size': 8,
                'xtick.major.width': 3, 'xtick.minor.width': 2, 'xtick.labelsize': 36})
        g = sns.FacetGrid(y_df, palette=palette, row="group", hue="group", aspect=7, height=2)
        g.map_dataframe(sns.kdeplot, x="response", hue='repeat', fill=True, alpha=0.25, log_scale=True, palette=palette, linewidth=0)
        g.map_dataframe(sns.kdeplot, x="response", hue='repeat', fill=False, alpha=1, log_scale=True, palette=palette, linewidth=2)
        g.refline(y=0, linewidth=1, linestyle="-", color='#212121', clip_on=False)
        g.refline(x=gateline, color='#212121', clip_on=False)
        def glabel(x, color, label):
            ax = plt.gca()
            labels_idx = y_df['group'].unique()
            labels_conv = dict(zip(labels_idx, group_labels))
            label = labels_conv[float(label)]
            ax.text(0, 0.02, label, color='#212121',
                fontsize=36, ha="left", transform=ax.transAxes)
        def annotate(data, **kws):
            ax = plt.gca()
            gl_tot = len(data)
            gl_high = np.sum(data['response'] > gateline)
            pct = np.around(100*(gl_high/gl_tot), 2)
            ax.text(1, 0.02, f"{pct}%", color='#212121',
                fontsize=36, ha="right", transform=ax.transAxes)
        g.map_dataframe(annotate)
        g.map(glabel, "group")
        g.fig.subplots_adjust(hspace=0)
        g.set_titles("")
        g.set(yticks=[], ylabel='', xlim=(0.5, 5e6))
        g.set_xlabels(xlabel, fontsize=48, fontweight='bold')
        # g.fig.supylabel('Probability Density', fontsize=36, fontweight='bold')
        g.despine(left=True, bottom=True)
        g.fig.savefig(save_path, dpi=100, pad_inches=0.1, bbox_inches='tight', transparent=False)
        plt.close('all')
        plt.close('fig')


def plot_obj_space_per_gen(root_dir):
    def task_func(save_dir, obj_scores, j):
        obj_gen = np.array(ast.literal_eval(obj_scores[j]))
        df = pd.DataFrame(data=obj_gen, columns=['obj1', 'obj2'])
        fig_path = os.path.join(save_dir, f'{j}.png')
        with plt.style.context(('seaborn-whitegrid', custom_styles)):
            fig, ax = plt.subplots(figsize=(24, 16))
            sns.scatterplot(data=df, x='obj1', y='obj2',
                edgecolor='#212121', facecolor='#648FFF', alpha=0.8, linewidth=2, s=1000)
            ax.set_xlabel('Response Performance')
            ax.set_ylabel('Network Simplicity')
            ax.set_xlim(0, 21)
            ax.set_ylim(-4, 2)
            fig.tight_layout()
            fig.canvas.draw()
            fig.savefig(fig_path, pad_inches=0.3, dpi=100, bbox_inches='tight', transparent=False)
            plt.cla()
            plt.clf()
            plt.close('all')
            plt.close('fig')
    for i, csvf in enumerate(natsorted([f for f in os.listdir(root_dir) if f.endswith('.csv')])):
        csvf = os.path.join(root_dir, csvf)
        run_dir = os.path.join(root_dir, str(i))
        print(run_dir)
        setup_dirs(run_dir)
        data = pd.read_csv(csvf)
        obj_scores = data['obj_scores'].values
        Parallel(n_jobs=os.cpu_count())(
            delayed(task_func)(run_dir, obj_scores, j)
            for j in range(len(obj_scores))
        )


def plot_3node_network(save_path, params):
    def calc_arrow_xy(xy_tail, xy_head, tail_pad, head_pad, shift=0):
        # calculate xy coordinates
        x1, y1 = xy_tail
        x2, y2 = xy_head
        dx = x2 - x1
        dy = y2 - y1
        new_x1 = x1 + dx*tail_pad
        new_y1 = y1 + dy*tail_pad
        new_x2 = x1 + dx*(1-head_pad)
        new_y2 = y1 + dy*(1-head_pad)
        if dy > 0:
            new_x1 -= shift
            new_x2 -= shift
        elif dy < 0:
            new_x1 += shift
            new_x2 += shift
        if dx > 0:
            new_y1 += shift
            new_y2 += shift
        elif dx < 0:
            new_y1 -= shift
            new_y2 -= shift
        xy_tail = (new_x1, new_y1)
        xy_head = (new_x2, new_y2)
        return xy_tail, xy_head

    def calc_arrow_fc(value):
        # colormap based on param value
        if value >= 1:
            fc = '#225888'
        elif value > 0 and value < 1:
            fc = '#c1d8ed'
        elif value == 0:
            fc = '#FFFFFF00'
        elif value > -1 and value < 0:
            fc = '#ecc3c1'
        elif value <= -1:
            fc = '#862722'
        return fc

    def calc_arrow_style(value):
        # repression vs activation arrow style
        if value < 0:
            arrow_style = mpatches.ArrowStyle("|-|, widthA=0, widthB=20")
        else:
            arrow_style = mpatches.ArrowStyle("-|>, head_length=20, head_width=20")
        return arrow_style

    def make_arrow(value, xy1=None, xy2=None, tail_pad=0.2, head_pad=0.4, shift=0, path=None):
        arrow_color = calc_arrow_fc(value)
        arrow_style = calc_arrow_style(value)
        if path is not None:
            arrow_patch = mpatches.FancyArrowPatch(
                path=path, color=arrow_color, arrowstyle=arrow_style, lw=10)
        else:
            arrow_tail, arrow_head = calc_arrow_xy(xy1, xy2, tail_pad, head_pad, shift)
            arrow_patch = mpatches.FancyArrowPatch(
                arrow_tail, arrow_head, color=arrow_color, arrowstyle=arrow_style, lw=10)
        return arrow_patch

    params = params.reshape(3, -1)
    kXr = params[:, 0]
    kXu = params[:, 1]
    params[:, 1] = np.where(kXr*kXu > 0, -kXu, kXu)
    [
        kAr, kAu, kAA, kAB, kAC, kAAt, kABt, kACt,
        kBr, kBu, kBA, kBB, kBC, kBAt, kBBt, kBCt,
        kCr, kCu, kCA, kCB, kCC, kCAt, kCBt, kCCt
    ] = params.ravel()
    with plt.style.context((custom_styles)):
        fig, ax = plt.subplots(figsize=(20, 20))
        with sns.axes_style("white"):
            A_ind = mpatches.Circle((0.55, 0.95), radius=0.02, fc='#648FFF', lw=0)
            A_rev = mpatches.Circle((0.85, 0.65), radius=0.02, fc='#212121', lw=0)
            A_com = mpatches.Circle((0.7, 0.8), radius=0.06, fc='#785EF0', lw=0)
            B_ind = mpatches.Circle((0.3, 0.29), radius=0.02, fc='#648FFF', lw=0)
            B_rev = mpatches.Circle((0.3, 0.71), radius=0.02, fc='#212121', lw=0)
            B_com = mpatches.Circle((0.3, 0.5), radius=0.06, fc='#FE6100', lw=0)
            C_ind = mpatches.Circle((0.55, 0.05), radius=0.02, fc='#648FFF', lw=0)
            C_rev = mpatches.Circle((0.85, 0.35), radius=0.02, fc='#212121', lw=0)
            C_com = mpatches.Circle((0.7, 0.2), radius=0.06, fc='#DC267F', lw=0)
            nodes = [A_ind, A_rev, A_com, B_ind, B_rev, B_com, C_ind, C_rev, C_com]
            A_kAu = make_arrow(kAu, (0.55, 0.95), (0.7, 0.8))
            A_kAr = make_arrow(kAr, (0.85, 0.65), (0.7, 0.8))
            B_kBu = make_arrow(kBu, (0.3, 0.29), (0.3, 0.5))
            B_kBr = make_arrow(kBr, (0.3, 0.71), (0.3, 0.5))
            C_kCu = make_arrow(kCu, (0.55, 0.05), (0.7, 0.2))
            C_kCr = make_arrow(kCr, (0.85, 0.35), (0.7, 0.2))
            A_kAB = make_arrow(kAB, (0.3, 0.5), (0.7, 0.8), 0.2, 0.2, 0.02)
            A_kAC = make_arrow(kAC, (0.7, 0.2), (0.7, 0.8), 0.2, 0.2, 0.02)
            B_kBA = make_arrow(kBA, (0.7, 0.8), (0.3, 0.5), 0.2, 0.2, 0.02)
            B_kBC = make_arrow(kBC, (0.7, 0.2), (0.3, 0.5), 0.2, 0.2, 0.02)
            C_kCA = make_arrow(kCA, (0.7, 0.8), (0.7, 0.2), 0.2, 0.2, 0.02)
            C_kCB = make_arrow(kCB, (0.3, 0.5), (0.7, 0.2), 0.2, 0.2, 0.02)
            AA_path = mpath.Path(
                [(0.7, 0.88), (0.7, 0.95), (0.85, 0.95), (0.85, 0.8), (0.78, 0.8)],
                [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.LINETO]
            )
            BB_path = mpath.Path(
                [(0.245, 0.445), (0.195, 0.395), (0.09, 0.5), (0.195, 0.605), (0.245, 0.555)],
                [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.LINETO]
            )
            CC_path = mpath.Path(
                [(0.78, 0.2), (0.85, 0.2), (0.85, 0.05), (0.7, 0.05), (0.7, 0.12)],
                [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.LINETO]
            )
            A_kAA = make_arrow(kAA, path=AA_path)
            B_kBB = make_arrow(kBB, path=BB_path)
            C_kCC = make_arrow(kCC, path=CC_path)
            edges = [
                A_kAu, A_kAr, A_kAA, A_kAB, A_kAC,
                B_kBu, B_kBr, B_kBA, B_kBB, B_kBC,
                C_kCu, C_kCr, C_kCA, C_kCB, C_kCC
            ]
            lgd_ind = mpatches.Circle((0.05, 0.95), radius=0.01, fc='#648FFF', lw=0)
            lgd_rev = mpatches.Circle((0.05, 0.90), radius=0.01, fc='#212121', lw=0)
            lgd_com = mpatches.Circle((0.05, 0.85), radius=0.015, fc='#FFFFFF', ec='#212121', lw=2)
            lgd_act1 = mpatches.Rectangle((0.16, 0.15), width=0.02, height=0.02, fc='#225888', lw=0)
            lgd_act01 = mpatches.Rectangle((0.12, 0.15), width=0.02, height=0.02, fc='#c1d8ed', lw=0)
            lgd_rep1 = mpatches.Rectangle((0.16, 0.05), width=0.02, height=0.02, fc='#862722', lw=0)
            lgd_rep01 = mpatches.Rectangle((0.12, 0.05), width=0.02, height=0.02, fc='#ecc3c1', lw=0)
            lgd = [lgd_ind, lgd_rev, lgd_com, lgd_act1, lgd_act01, lgd_rep1, lgd_rep01]
            for element in nodes + edges + lgd:
                ax.add_patch(element)
            ax.text(0.7, 0.8, 'X1', color='#212121', size=86, ha='center', va='center_baseline')
            ax.text(0.3, 0.5, 'X2', color='#212121', size=86, ha='center', va='center_baseline')
            ax.text(0.7, 0.2, 'X3', color='#212121', size=86, ha='center', va='center_baseline')
            ax.text(0.05, 0.85, 'X', color='#212121', size=24, ha='center', va='center_baseline')
            ax.text(0.075, 0.95, 'Induction Stimuli', color='#212121', size=64, ha='left', va='center_baseline')
            ax.text(0.075, 0.90, 'Reversion Process', color='#212121', size=64, ha='left', va='center_baseline')
            ax.text(0.075, 0.85, 'Cell Components', color='#212121', size=64, ha='left', va='center_baseline')
            ax.text(0.15, 0.20, 'Activation', color='#212121', size=64, ha='center', va='center_baseline')
            ax.text(0.15, 0.10, 'Repression', color='#212121', size=64, ha='center', va='center_baseline')
            ax.text(0.1, 0.16, 'Weak', color='#212121', size=42, ha='right', va='center_baseline')
            ax.text(0.2, 0.16, 'Strong', color='#212121', size=42, ha='left', va='center_baseline')
            ax.text(0.1, 0.06, 'Weak', color='#212121', size=42, ha='right', va='center_baseline')
            ax.text(0.2, 0.06, 'Strong', color='#212121', size=42, ha='left', va='center_baseline')
            sns.despine(fig=fig, ax=ax, left=True, bottom=True)
            ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(save_path, pad_inches=0.0, dpi=100, bbox_inches='tight', transparent=False)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')
