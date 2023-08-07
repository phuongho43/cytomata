import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib import font_manager
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from skimage import img_as_ubyte

from cytomata.process import revert_tu
from cytomata.utils import setup_dirs, custom_styles, custom_palette


def plot_cell_img(img, fname, save_dir, cmax, regions=None, centroids=None, sig_ann=False, t_unit=None, sb_microns=None):
    setup_dirs(save_dir)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(18, 12))
        axim = ax.imshow(img, cmap='turbo')
        axim.set_clim(0.0, cmax)
        if t_unit:
            t_text = 't = ' + fname + t_unit
            ax.text(0.05, 0.95, t_text, ha='left', va='center', color='white',
                fontsize=28, weight='bold', transform=ax.transAxes)
        if sb_microns is not None:
            fontprops = font_manager.FontProperties(size=56, weight='bold')
            asb = AnchoredSizeBar(ax.transData, 300, u'{}\u03bcm'.format(sb_microns),
                color='white', size_vertical=6, fontproperties=fontprops,
                loc='lower left', pad=0, borderpad=1, sep=5, frameon=False)
            ax.add_artist(asb)
        if sig_ann:
            w, h = img.shape
            ax.add_patch(mpatches.Rectangle((2, 2), w-7, h-7, linewidth=10, edgecolor='#1e88e5ff', facecolor='none'))
        ax.grid(False)
        ax.axis('off')
        cb = fig.colorbar(axim, pad=0.01, format='%.3f', extend='both', extendrect=True,
                ticks=np.linspace(np.min(img), cmax, 10))
        cb.outline.set_linewidth(0)
        fig.tight_layout(pad=0)
        if regions is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ax.contour(regions, linewidths=3, colors='w')
            if centroids is not None:
                for num, (y, x) in centroids.items():
                    ax.annotate(str(num), xy=(x, y),
                        xycoords='data', color='white', fontsize=36, ha='center', va='center_baseline')
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, fname + '.png'),
                    dpi=300, bbox_inches='tight', pad_inches=0)
        cell_img = img_as_ubyte(np.array(fig.canvas.renderer._renderer))
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')
        return cell_img


def plot_bkg_profile(img, bkg, fname, save_dir):
    setup_dirs(os.path.join(save_dir, 'debug'))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        bg_rows = np.argsort(np.var(img, axis=1))[-100:-1:10]
        row_i = np.random.choice(bg_rows.shape[0])
        bg_row = bg_rows[row_i]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(img[bg_row, :])
        ax.plot(bkg[bg_row, :])
        # ax.set_title(str(bg_row))
        bg_path = os.path.join(save_dir, 'debug', '{}.png'.format(fname))
        fig.savefig(bg_path, pad_inches=0, bbox_inches='tight', transparent=False, dpi=100)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')


def plot_class_group(y_df, save_path, group_labels, class_labels=[], x_var='group', y_var='response', h_var='class', ylabel='', xlabel='', ymin=None, ymax=None, lgd_loc='best', palette=custom_palette, rc=None, figsize=(24, 8)):
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(palette):
        if rc is not None:
            sns.set_context(rc=rc)
        fig, ax = plt.subplots(figsize=figsize)
        # y_df[y_var] = np.log10(y_df[y_var])
        dodge_0 = False
        dodge_1 = False
        if class_labels:
            dodge_0 = True
            dodge_1 = 0.4
        glen = 1 if len(group_labels)==0 else len(group_labels)
        clen = 1 if len(class_labels)==0 else len(class_labels)
        capsize = 0.015*(glen+1)*(clen+1)
        sns.stripplot(x=x_var, y=y_var, hue=h_var, data=y_df, ax=ax, palette=palette, dodge=dodge_0,
                      jitter=0.1, legend=True, size=42, linewidth=2, edgecolor='#212121',
                      alpha=0.8, zorder=0)
        sns.pointplot(x=x_var, y=y_var, hue=h_var, data=y_df, ax=ax, color='#212121', dodge=dodge_1,
                      estimator='mean', errorbar=('se', 1.96), markers='.', errwidth=10, join=False,
                      capsize=capsize, scale=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(group_labels)
        # ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
        #     ymin, ymax = y_df[y_var].min(), y_df[y_var].max()
        #     tick_range = np.arange(np.floor(ymin), np.ceil(ymax) + 1)
        #     ax.yaxis.set_ticks(tick_range)
        #     tick_range = np.arange(np.floor(ymin), np.ceil(ymax))
        #     ax.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(10 ** p, 10 ** (p + 1), 10)], minor=True)
        ymin_i, ymax_i = ax.get_ylim()
        if ymax is None:
            ymax = ymax_i
        if ymin is None:
            ymin = ymin_i
        ax.set_ylim(ymin, ymax)
        ax.tick_params(which='minor', length=8, width=2)
        ax.tick_params(which='major', length=12, width=4)
        handles, labels = ax.get_legend_handles_labels()
        if class_labels:
            ax.legend(handles, class_labels, loc=lgd_loc,
                markerscale=4, frameon=True, shadow=False, handletextpad=0.1, borderpad=0.1, labelspacing=0.2, handlelength=1)
        plt.savefig(save_path, pad_inches=0.2, dpi=300, transparent=False, bbox_inches='tight')
        plt.close()


def plot_uy(y_df, u_df, save_dir, yd_df=None, fname='y.png', style=False, logx=False, dpi=300, overlay_tu=False, markers=False, lgd_loc='best', xlabel='Time (s)', ylabel=r'$\mathbf{\Delta F/F_{0}}$', ulabel='BL', group_labels=None, ymin=None, ymax=None, order=None, palette=custom_palette):
    setup_dirs(save_dir)
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        if u_df is not None:
            fig, (ax0, ax) = plt.subplots(
                2, 1, sharex=True, figsize=(16, 10),
                gridspec_kw={'height_ratios': [1, 8]}
            )
            sns.lineplot(data=u_df, x="t", y="u", ax=ax0, errorbar=None, color='#648FFF')
            ax0.set_yticks([0, 1])
            ax0.set_ylabel(ulabel)
        else:
            fig, ax = plt.subplots(figsize=(18, 10))
        if 'h' in y_df.columns:
            sns.lineplot(data=y_df, x="t", y="y", ax=ax, hue='h', style=style, style_order=order, hue_order=order,
                    dashes=[(1, 0), (2, 2), (1, 0), (2, 1), (3, 3)], errorbar=None, palette=palette)
            if yd_df is not None:
                sns.scatterplot(data=yd_df, x="t", y="y", hue="h", style='h', edgecolor='#212121', linewidth=1, s=100, palette=palette, zorder=10)
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, group_labels, loc=lgd_loc,
                markerscale=4, frameon=True, shadow=False, handletextpad=0.2, borderpad=0.2, labelspacing=0.2, handlelength=1)
        else:
            sns.lineplot(data=y_df, x="t", y="y", ax=ax, estimator='mean', errorbar=("se", 1.96), color=palette[0], markers=markers)
        if overlay_tu and u_df is not None:
            ta_tb = revert_tu(u_df)
            for (ta, tb) in ta_tb:
                plt.axvspan(ta, tb, color='#648FFF', alpha=0.3, lw=0, zorder=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.locator_params(axis='x', nbins=10)
        ax.locator_params(axis='y', nbins=8)
        ymin_i, ymax_i = ax.get_ylim()
        if ymax is None:
            ymax = ymax_i
        if ymin is None:
            ymin = ymin_i
        ax.set_ylim(ymin, ymax)
        if logx:
            ax.set_xscale('log')
            ax.set_xticks(logx)
            plt.xticks(rotation=45)
            ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, fname), pad_inches=0.3, dpi=300, bbox_inches='tight', transparent=False)
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
        g.fig.savefig(save_path, dpi=300, pad_inches=0.1, bbox_inches='tight', transparent=False)
        plt.close('all')
        plt.close('fig')


def plot_netw_fitn(save_dir, n_fit, y_fit, t_conc, y_conc, params):
    def make_arrow(x1, y1, x2, y2, value, shift=0.02, inhib=False, thr=0):
        width = 0
        alpha = 1
        if abs(value) >= 10:
            width = 0.015
        elif abs(value) >= 1:
            width = 0.01
        elif abs(value) >= 0.1:
            width = 0.005
        elif abs(value) > 0:
            width = 0.0025
        elif abs(value) == 0:
            width = 0
        head_width = 3*width
        head_length = 1*head_width
        overhang = 0.1
        if value < 0:
            if inhib:
                head_width = 5*width
                head_length = 0.02*head_width
                overhang = -10
            else:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                shift = -shift
        x = (x2 - x1)*0.2 + x1
        y = (y2 - y1)*0.2 + y1
        dx = (x2 - x1)*0.6
        dy = (y2 - y1)*0.6
        if dy > 0:
            x -= shift
        elif dy < 0:
            x += shift
        if dx > 0:
            y += shift
        elif dx < 0:
            y-= shift
        if thr > 0.5:
            alpha = 0.25
        elif thr > 0.25:
            alpha = 0.5
        elif thr > 0:
            alpha = 0.75
        arrow = {'x': x, 'y': y, 'dx': dx, 'dy': dy, 'width': width, 'head_width': head_width, 'head_length': head_length, 'overhang': overhang, 'alpha': alpha}
        return arrow
    [kra, krb, krc,
     kua, kub, kuc,
     kta, ktb, ktc,
     kba, kab, kac,
     kca, kcb, kbc,
    ] = params
    kua = -kua if kra > 0 else kua
    kub = -kub if krb > 0 else kub
    kuc = -kuc if krc > 0 else kuc
    with plt.style.context((custom_styles)):
        fig = plt.figure(figsize=(40, 20))
        gs = fig.add_gridspec(2, 2)
        with sns.axes_style("white"):
            ax0 = fig.add_subplot(gs[:, 0])
            Ai_patch = mpatches.Circle((0.5, 0.8), clip_on=False, radius=0.04, fc='#BCB1F3', ec='#212121', lw=4)
            Aa_patch = mpatches.Circle((0.9, 0.8), clip_on=False, radius=0.04, fc='#785EF0', ec='#212121', lw=4)
            Bi_patch = mpatches.Circle((0.1, 0.5), clip_on=False, radius=0.04, fc='#FFB78E', ec='#212121', lw=4)
            Ba_patch = mpatches.Circle((0.5, 0.5), clip_on=False, radius=0.04, fc='#FE6100', ec='#212121', lw=4)
            Ci_patch = mpatches.Circle((0.5, 0.2), clip_on=False, radius=0.04, fc='#F3ACCA', ec='#212121', lw=4)
            Ca_patch = mpatches.Circle((0.9, 0.2), clip_on=False, radius=0.04, fc='#DC267F', ec='#212121', lw=4)
            kua_arrow = make_arrow(0.5, 0.8, 0.9, 0.8, value=kua, shift=0.02)
            kua_patch = mpatches.FancyArrow(**kua_arrow, fc='#648FFF', length_includes_head=True)
            kra_arrow = make_arrow(0.5, 0.8, 0.9, 0.8, value=kra, shift=-0.02)
            kra_patch = mpatches.FancyArrow(**kra_arrow, fc='#424242', length_includes_head=True)
            kub_arrow = make_arrow(0.1, 0.5, 0.5, 0.5, value=kub, shift=0.02)
            kub_patch = mpatches.FancyArrow(**kub_arrow, fc='#648FFF', length_includes_head=True)
            krb_arrow = make_arrow(0.1, 0.5, 0.5, 0.5, value=krb, shift=-0.02)
            krb_patch = mpatches.FancyArrow(**krb_arrow, fc='#424242', length_includes_head=True)
            kuc_arrow = make_arrow(0.5, 0.2, 0.9, 0.2, value=kuc, shift=0.02)
            kuc_patch = mpatches.FancyArrow(**kuc_arrow, fc='#648FFF', length_includes_head=True)
            krc_arrow = make_arrow(0.5, 0.2, 0.9, 0.2, value=krc, shift=-0.02)
            krc_patch = mpatches.FancyArrow(**krc_arrow, fc='#424242', length_includes_head=True)
            kab_arrow = make_arrow(0.9, 0.8, 0.5, 0.5, value=kab, shift=0.015, inhib=True, thr=kta)
            kab_patch = mpatches.FancyArrow(**kab_arrow, fc='#785EF0', length_includes_head=True)
            kba_arrow = make_arrow(0.5, 0.5, 0.9, 0.8, value=kba, shift=0.015, inhib=True, thr=ktb)
            kba_patch = mpatches.FancyArrow(**kba_arrow, fc='#FE6100', length_includes_head=True)
            kac_arrow = make_arrow(0.9, 0.8, 0.9, 0.2, value=kac, shift=0.02, inhib=True, thr=kta)
            kac_patch = mpatches.FancyArrow(**kac_arrow, fc='#785EF0', length_includes_head=True)
            kca_arrow = make_arrow(0.9, 0.2, 0.9, 0.8, value=kca, shift=0.02, inhib=True, thr=ktc)
            kca_patch = mpatches.FancyArrow(**kca_arrow, fc='#DC267F', length_includes_head=True)
            kbc_arrow = make_arrow(0.5, 0.5, 0.9, 0.2, value=kbc, shift=0.015, inhib=True, thr=ktb)
            kbc_patch = mpatches.FancyArrow(**kbc_arrow, fc='#FE6100', length_includes_head=True)
            kcb_arrow = make_arrow(0.9, 0.2, 0.5, 0.5, value=kcb, shift=0.015, inhib=True, thr=ktc)
            kcb_patch = mpatches.FancyArrow(**kcb_arrow, fc='#DC267F', length_includes_head=True)
            variables = [Ai_patch, Aa_patch, Bi_patch, Ba_patch, Ci_patch, Ca_patch]
            parameters = [kua_patch, kra_patch, kub_patch, krb_patch, kuc_patch, krc_patch, kab_patch, kba_patch, kac_patch, kca_patch, kbc_patch, kcb_patch]
            elements = variables + parameters
            for element in elements:
                ax0.add_patch(element)
            ax0.text(0.5, 0.8, r'$\mathtt{A_{i}}$', color='#212121', size=52, ha='center', va='center')
            ax0.text(0.9, 0.8, r'$\mathtt{A_{a}}$', color='#FAFAFA', size=52, ha='center', va='center')
            ax0.text(0.1, 0.5, r'$\mathtt{B_{i}}$', color='#212121', size=52, ha='center', va='center')
            ax0.text(0.5, 0.5, r'$\mathtt{B_{a}}$', color='#FAFAFA', size=52, ha='center', va='center')
            ax0.text(0.5, 0.2, r'$\mathtt{C_{i}}$', color='#212121', size=52, ha='center', va='center')
            ax0.text(0.9, 0.2, r'$\mathtt{C_{a}}$', color='#FAFAFA', size=52, ha='center', va='center')
            legend_items = [
                Line2D([0], [0], color='#648FFF', lw=12, label='Induction'),
                Line2D([0], [0], color='#424242', lw=12, label='Reversion'),
                Line2D([0], [0], color='#785EF0', lw=12, label=r'$\mathdefault{[A_a]\ Interaction}$'),
                Line2D([0], [0], color='#FE6100', lw=12, label=r'$\mathdefault{[B_a]\ Interaction}$'),
                Line2D([0], [0], color='#DC267F', lw=12, label=r'$\mathdefault{[C_a]\ Interaction}$')]
            ax0.legend(handles=legend_items, loc='upper left')
            sns.despine(fig=fig, ax=ax0, left=True, bottom=True)
            ax0.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        with sns.axes_style("whitegrid"):
            ax1 = fig.add_subplot(gs[0, 1])
            ax1.plot(n_fit, y_fit, color='#34495E')
            ax1.set_ylabel('Fitness')
            ax1.set_xlabel('Generation')
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.plot(t_conc, y_conc, color='#DC267F')
            ax2.set_ylabel(r'$\mathdefault{[C_a]}$', rotation=0, labelpad=72)
            ax2.set_xlabel('Time')
            ax2.set_ylim([0, 1])
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, f'{n_fit[-1]}.png'), pad_inches=0.1, dpi=100, bbox_inches='tight', transparent=False)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')



def plot_model(save_dir, params):
    def make_arrow(x1, y1, x2, y2, value, shift=0.02, inhib=False, thr=0):
        width = 0
        alpha = 1
        if abs(value) >= 10:
            width = 0.02
        elif abs(value) >= 1:
            width = 0.01
        elif abs(value) > 0:
            width = 0.005
        elif abs(value) == 0:
            width = 0
            alpha = 0
        head_width = 3*width
        head_length = 1*head_width
        overhang = 0.1
        if value < 0:
            if inhib:
                head_width = 5*width
                head_length = 0.02*head_width
                overhang = -10
            else:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                shift = -shift
        x = (x2 - x1)*0.2 + x1
        y = (y2 - y1)*0.2 + y1
        dx = (x2 - x1)*0.6
        dy = (y2 - y1)*0.6
        if dy > 0:
            x -= shift
        elif dy < 0:
            x += shift
        if dx > 0:
            y += shift
        elif dx < 0:
            y-= shift
        if thr > 0.5:
            alpha = 0.25
        elif thr > 0.25:
            alpha = 0.5
        elif thr > 0:
            alpha = 0.75
        arrow = {'x': x, 'y': y, 'dx': dx, 'dy': dy, 'width': width, 'head_width': head_width, 'head_length': head_length, 'overhang': overhang, 'alpha': alpha}
        return arrow
    [kra, krb, krc,
     kua, kub, kuc,
     kta, ktb, ktc,
     kba, kab, kac,
     kca, kcb, kbc,
    ] = params
    kua = -kua if kra > 0 else kua
    kub = -kub if krb > 0 else kub
    kuc = -kuc if krc > 0 else kuc
    with plt.style.context((custom_styles)):
        fig, ax = plt.subplots(figsize=(20, 20))
        with sns.axes_style("white"):
            Ai_patch = mpatches.Circle((0.5, 0.8), clip_on=False, radius=0.06, fc='#BCB1F3', ec='#212121', lw=4)
            Aa_patch = mpatches.Circle((0.9, 0.8), clip_on=False, radius=0.06, fc='#785EF0', ec='#212121', lw=4)
            Bi_patch = mpatches.Circle((0.1, 0.5), clip_on=False, radius=0.06, fc='#FFB78E', ec='#212121', lw=4)
            Ba_patch = mpatches.Circle((0.5, 0.5), clip_on=False, radius=0.06, fc='#FE6100', ec='#212121', lw=4)
            Ci_patch = mpatches.Circle((0.5, 0.2), clip_on=False, radius=0.06, fc='#F3ACCA', ec='#212121', lw=4)
            Ca_patch = mpatches.Circle((0.9, 0.2), clip_on=False, radius=0.06, fc='#DC267F', ec='#212121', lw=4)
            kua_arrow = make_arrow(0.5, 0.8, 0.9, 0.8, value=kua, shift=0.02)
            kua_patch = mpatches.FancyArrow(**kua_arrow, fc='#648FFF', length_includes_head=True)
            kra_arrow = make_arrow(0.5, 0.8, 0.9, 0.8, value=kra, shift=-0.02)
            kra_patch = mpatches.FancyArrow(**kra_arrow, fc='#424242', length_includes_head=True)
            kub_arrow = make_arrow(0.1, 0.5, 0.5, 0.5, value=kub, shift=0.02)
            kub_patch = mpatches.FancyArrow(**kub_arrow, fc='#648FFF', length_includes_head=True)
            krb_arrow = make_arrow(0.1, 0.5, 0.5, 0.5, value=krb, shift=-0.02)
            krb_patch = mpatches.FancyArrow(**krb_arrow, fc='#424242', length_includes_head=True)
            kuc_arrow = make_arrow(0.5, 0.2, 0.9, 0.2, value=kuc, shift=0.02)
            kuc_patch = mpatches.FancyArrow(**kuc_arrow, fc='#648FFF', length_includes_head=True)
            krc_arrow = make_arrow(0.5, 0.2, 0.9, 0.2, value=krc, shift=-0.02)
            krc_patch = mpatches.FancyArrow(**krc_arrow, fc='#424242', length_includes_head=True)
            kab_arrow = make_arrow(0.9, 0.8, 0.5, 0.5, value=kab, shift=0.015, inhib=True, thr=kta)
            kab_patch = mpatches.FancyArrow(**kab_arrow, fc='#785EF0', length_includes_head=True)
            kba_arrow = make_arrow(0.5, 0.5, 0.9, 0.8, value=kba, shift=0.015, inhib=True, thr=ktb)
            kba_patch = mpatches.FancyArrow(**kba_arrow, fc='#FE6100', length_includes_head=True)
            kac_arrow = make_arrow(0.9, 0.8, 0.9, 0.2, value=kac, shift=0.02, inhib=True, thr=kta)
            kac_patch = mpatches.FancyArrow(**kac_arrow, fc='#785EF0', length_includes_head=True)
            kca_arrow = make_arrow(0.9, 0.2, 0.9, 0.8, value=kca, shift=0.02, inhib=True, thr=ktc)
            kca_patch = mpatches.FancyArrow(**kca_arrow, fc='#DC267F', length_includes_head=True)
            kbc_arrow = make_arrow(0.5, 0.5, 0.9, 0.2, value=kbc, shift=0.015, inhib=True, thr=ktb)
            kbc_patch = mpatches.FancyArrow(**kbc_arrow, fc='#FE6100', length_includes_head=True)
            kcb_arrow = make_arrow(0.9, 0.2, 0.5, 0.5, value=kcb, shift=0.015, inhib=True, thr=ktc)
            kcb_patch = mpatches.FancyArrow(**kcb_arrow, fc='#DC267F', length_includes_head=True)
            variables = [Ai_patch, Aa_patch, Bi_patch, Ba_patch, Ci_patch, Ca_patch]
            parameters = [kua_patch, kra_patch, kub_patch, krb_patch, kuc_patch, krc_patch, kab_patch, kba_patch, kac_patch, kca_patch, kbc_patch, kcb_patch]
            elements = variables + parameters
            for element in elements:
                ax.add_patch(element)
            ax.text(0.5, 0.8, r'$\mathtt{A_{i}}$', color='#212121', size=86, ha='center', va='center')
            ax.text(0.9, 0.8, r'$\mathtt{A_{a}}$', color='#FAFAFA', size=86, ha='center', va='center')
            ax.text(0.1, 0.5, r'$\mathtt{B_{i}}$', color='#212121', size=86, ha='center', va='center')
            ax.text(0.5, 0.5, r'$\mathtt{B_{a}}$', color='#FAFAFA', size=86, ha='center', va='center')
            ax.text(0.5, 0.2, r'$\mathtt{C_{i}}$', color='#212121', size=86, ha='center', va='center')
            ax.text(0.9, 0.2, r'$\mathtt{C_{a}}$', color='#FAFAFA', size=86, ha='center', va='center')
            legend_items = [
                Line2D([0], [0], color='#648FFF', lw=12, label='Induction'),
                Line2D([0], [0], color='#424242', lw=12, label='Reversion'),
                Line2D([0], [0], color='#785EF0', lw=12, label=r'$\mathdefault{[A_a]\ Effect}$'),
                Line2D([0], [0], color='#FE6100', lw=12, label=r'$\mathdefault{[B_a]\ Effect}$'),
                Line2D([0], [0], color='#DC267F', lw=12, label=r'$\mathdefault{[C_a]\ Effect}$')]
            # ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(0.05, 0.85))
            sns.despine(fig=fig, ax=ax, left=True, bottom=True)
            ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'model.png'), pad_inches=0.0, dpi=300, bbox_inches='tight', transparent=False)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close('fig')
