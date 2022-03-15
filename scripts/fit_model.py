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
from natsort import natsorted

from cytomata.model import sim_itrans, sim_idimer, sim_express, sim_fresca
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def prep_iLID_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    yd = ydf['y'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    ydf = interp1d(td, yd, kind='linear')
    y = np.array([ydf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    return t, y, u


def fit_idimer(t1, y1, u1, t2, y2, u2, save_dir):
    y01 = [1, 0, 1, 0]
    uf1 = interp1d(t1, u1, kind='nearest')
    y02 = [1, 0, 1, 0]
    uf2 = interp1d(t2, u2, kind='nearest')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        params1 = {'kl': params['kl'].value, 'kd': params['kd1'].value, 'kf': params['kf'].value, 'kr': params['kr'].value}
        tm1, ym1 = sim_idimer(t1, y01, uf1, params1)
        params2 = {'kl': params['kl'].value, 'kd': params['kd2'].value, 'kf': params['kf'].value, 'kr': params['kr'].value}
        tm2, ym2 = sim_idimer(t2, y02, uf2, params2)
        res1 = np.sum((ym1[-1] - y1)**2)
        res2 = np.sum((ym2[-1] - y2)**2)
        return  res1 + res2
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
    params.add('kl', value=0.1, min=0.0, max=100.0)
    params.add('kd1', value=0.1, min=0.0, max=100.0)
    params.add('kd2', value=0.1, min=0.0, max=100.0)
    params.add('kf', value=0.1, min=0.0, max=100.0)
    params.add('kr', value=0.1, min=0.0, max=100.0)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='powell',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-6
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    print(opt_params)
    with open(os.path.join(save_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    params1 = {'kl': opt_params['kl'], 'kd': opt_params['kd1'], 'kf': opt_params['kf'], 'kr': opt_params['kr']}
    tm1, ym1 = sim_idimer(t1, y01, uf1, params1)
    params2 = {'kl': opt_params['kl'], 'kd': opt_params['kd2'], 'kf': opt_params['kf'], 'kr': opt_params['kr']}
    tm2, ym2 = sim_idimer(t2, y02, uf2, params2)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(24, 16), gridspec_kw={'height_ratios': [1, 8, 8]})
        ax0.plot(t1, u1)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t1, y1, color='#DC267F', label='Data (iLIDfast)', linestyle='dashed', linewidth=3)
        ax1.plot(tm1, ym1[-1], color='#648FFF', label='Model (iLIDfast)', alpha=0.6)
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        ax2.plot(t2, y2, color='#DC267F', label='Data (iLIDslow)', linestyle='dashed', linewidth=3)
        ax2.plot(tm2, ym2[-1], color='#648FFF', label='Model (iLIDslow', alpha=0.6)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('AU')
        ax2.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


if __name__ == '__main__':
    # root_dir1 = '/home/phuong/data/ILID/ddFP/RA-27V/20200921-B3-sspBu_RA-27V_spike/results/'
    # root_dir2 = '/home/phuong/data/ILID/ddFP/RA-16I/B3-sspBu_RA-16I_BL1-1s/mCherry-results/'
    # save_dir = '/home/phuong/data/ILID/ddFP/'
    # u_csv1 = os.path.join(root_dir1, 'u.csv')
    # u_csv2 = os.path.join(root_dir2, 'u.csv')
    # y_csv1 = os.path.join(root_dir1, 'y.csv')
    # y_csv2 = os.path.join(root_dir2, 'y.csv')
    # y_data1 = pd.read_csv(y_csv1)
    # y_data2 = pd.read_csv(y_csv2)
    # t1 = y_data1['t_ave'].values
    # t2 = y_data2['t_ave'].values
    # y1 = y_data1['y_ave'].values
    # y2 = y_data2['y_ave'].values
    # y1 = (y1 - y1[0])
    # y2 = (y2 - y2[0])
    # yf1 = interp1d(t1, y1, kind='linear')
    # yf2 = interp1d(t2, y2, kind='linear')
    # t1 = np.around(np.arange(t1[0], t1[-1], 0.1), 1)
    # t2 = np.around(np.arange(t2[0], t2[-1], 0.1), 1)
    # y1 = np.array([yf1(ti) for ti in t1])
    # y2 = np.array([yf2(ti) for ti in t2])
    # u_data1 = pd.read_csv(u_csv1)
    # u_data2 = pd.read_csv(u_csv2)
    # tu1 = u_data1['tu_ave'].values
    # tu2 = u_data2['tu_ave'].values
    # u1 = u_data1['u_ave'].values
    # u2 = u_data2['u_ave'].values
    # uf1 = interp1d(tu1, u1, kind='nearest', bounds_error=False, fill_value=0)
    # uf2 = interp1d(tu2, u2, kind='nearest', bounds_error=False, fill_value=0)
    # u1 = np.array([uf1(ti) for ti in t1])
    # u2 = np.array([uf2(ti) for ti in t2])
    # fit_idimer(t1, y1, u1, t2, y2, u2, save_dir)


    save_dir = '/home/phuong/data/ILID/ddFP/'
    t = np.arange(0.0, 300.0)
    u = np.zeros_like(t)
    u[60:120:2] = 1
    u[120:240:30] = 1
    uf = interp1d(t, u, kind='nearest')
    u1 = np.array([uf(ti) for ti in t])
    y0 = [10, 1, 0, 0, 1, 0, 0]
    paramsf = '/home/phuong/data/ILID/ddFP/opt_params.json'
    with open(paramsf) as f:
          params = json.load(f)
    tm, ym = sim_fresca(t, y0, uf, params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(24, 16), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u1, linewidth=2)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(tm, ym[-2], color='#4CAF50', label='iLIDfast-sspB')
        ax1.plot(tm, ym[-1], color='#F44336', label='iLIDslow-sspB')
        ax1.set_ylabel('AU')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylim([0, 1])
        ax1.legend(loc='upper right')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(save_dir, 'sim_10fast-1sspB-1slow_BL-1s-2s-t60-t120_BL-1s-30s-t120-t240.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)