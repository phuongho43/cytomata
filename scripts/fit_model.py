import os
import sys
import time
import json
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps
from natsort import natsorted

from cytomata.model import sim_idimer, sim_idissoc, sim_ifate
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def fit_idimer(td, yd, ud, save_dir):
    y0 = [10, 0, 10, 1, 0]
    uf = interp1d(td, ud, kind='nearest')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_idimer(td, y0, uf, params)
        res = np.sum(((ym[-1] + ym[-2]) - yd)**2)
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    params = lm.Parameters()
    params.add('kl', value=0.0, min=0.0, max=10.0)
    params.add('kd', value=0.0, min=0.0, max=10.0)
    params.add('kif', value=0.0, min=0.0, max=10.0)
    params.add('kaf', value=0.0, min=0.0, max=10.0)
    params.add('kr', value=0.0, min=0.0, max=10.0)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-6
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    print(opt_params)
    with open(os.path.join(save_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_idimer(td, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(24, 16), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(td, ud)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(td, yd, color='#785EF0', label='Data', linestyle='dashed', linewidth=3)
        ax1.plot(tm, ym[-1]+ym[-2], color='#648FFF', label='Model', alpha=0.6)
        ax1.set_ylabel('AU')
        ax1.set_xlabel('Time (s)')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def fit_idissoc(td, yd, ud, save_dir):
    y0 = [np.min(yd), np.min(yd), yd[0]]
    uf = interp1d(td, ud, kind='nearest')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_idissoc(td, y0, uf, params)
        res = np.sum((ym[-1] - yd)**2)
        return  res
    def opt_iter(params, iter, res):
        nonlocal min_res, best_params, iter_t
        clear_screen()
        ti = time.time()
        print('seconds/iter:', str(ti - iter_t))
        iter_t = ti
        print('Iter: {} | Res: {}'.format(iter, res))
        print(params.valuesdict())
        if res < min_res:
            min_res = res
            best_params = params.valuesdict()
        print('Best so far:')
        print('Res:', str(min_res))
        print(best_params)
    params = lm.Parameters()
    params.add('kl', value=0.0, min=0.0, max=10.0)
    params.add('kd', value=0.0, min=0.0, max=10.0)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='nelder',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-6
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    print(opt_params)
    with open(os.path.join(save_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_idissoc(td, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(24, 16), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(td, ud)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(td, yd, color='#785EF0', label='Data', linestyle='dashed', linewidth=3)
        ax1.plot(tm, ym[-1], color='#648FFF', label='Model', alpha=0.6)
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


if __name__ == '__main__':
    # ## iDimer (iLID) ###
    # root_dir = '/home/phuong/data/1-ifate/2-modeling/iLID27V/'
    # y_csv = os.path.join(root_dir, 'y.csv')
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y'].values
    # u = y_data['u'].values
    # fit_idimer(t, y, u, root_dir)


    # ### iDissoc (LOV) ###
    # root_dir = '/home/phuong/data/1-ifate/2-modeling/LOV27V/'
    # y_csv = os.path.join(root_dir, 'y.csv')
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y'].values
    # u = y_data['u'].values
    # fit_idissoc(t, y, u, root_dir)


    save_dir = '/home/phuong/data/1-ifate/2-modeling/'
    t = np.arange(0.0, 300.0)
    y0 = [1, 1, 1, 1, 1, 1, 1, 1]
    params = {
        'kl1': 0.3540261525923416, 'kd1': 0.34266130365617986,
        'kl2': 0.6613735116279074, 'kd2': 0.15594925769986223,
        'kl3': 0.2697506295580904, 'kd3': 0.00564155126989596
    }
    periods = list(range(120, 0, -1))
    AB_AUCs = []
    BCD_AUCs = []
    for period in periods:
        u = np.zeros_like(t)
        u[0:300:period] = 1
        uf = interp1d(t, u, kind='nearest')
        u1 = np.array([uf(ti) for ti in t])
        tm, ym = sim_ifate(t, y0, uf, params)
        AB_AUCs.append(simps(ym[4], tm))
        BCD_AUCs.append(simps(ym[-1], tm))
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, ax = plt.subplots(figsize=(24, 16))
        ax.plot(periods, BCD_AUCs, color='#785EF0', label='sparse channel')
        ax.plot(periods, AB_AUCs, color='#DC267F', label='dense channel')
        ax.set_ylabel('AUC')
        ax.set_xlabel('BL Period (s)')
        # ax1.set_ylim([0, 1.1])
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'period-AUCs.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


    # save_dir = '/home/phuong/data/1-ifate/2-modeling/'
    # t = np.arange(0.0, 300.0)
    # y0 = [1, 1, 1, 1, 1, 1, 1, 1]
    # params = {
    #     'kl1': 0.3540261525923416, 'kd1': 0.34266130365617986,
    #     'kl2': 0.6613735116279074, 'kd2': 0.15594925769986223,
    #     'kl3': 0.2697506295580904, 'kd3': 0.00564155126989596
    # }
    # u = np.zeros_like(t)
    # u[0:300:120] = 1
    # uf = interp1d(t, u, kind='nearest')
    # u1 = np.array([uf(ti) for ti in t])
    # tm, ym = sim_ifate(t, y0, uf, params)
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(24, 16), gridspec_kw={'height_ratios': [1, 8]})
    #     ax0.plot(t, u1, linewidth=2)
    #     ax0.set_yticks([0, 1])
    #     ax0.set_ylabel('BL')
    #     ax1.plot(tm, ym[-1], color='#785EF0', label='sparse channel')
    #     ax1.plot(tm, ym[4], color='#DC267F', label='dense channel')
    #     ax1.set_ylabel('AU')
    #     ax1.set_xlabel('Time (s)')
    #     # ax1.set_ylim([0, 1.1])
    #     ax1.legend(loc='upper right')
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     fig.savefig(os.path.join(save_dir, 'sim_ifate_1s-120s.png'),
    #         dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
    #     plt.close(fig)