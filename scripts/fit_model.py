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

from cytomata.model import sim_itranslo, sim_idimer, sim_iexpress, sim_ssl, sim_CaM_M13, sim_ilid, sim_fresca
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette, rescale


def prep_itranslo_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    ycd = ydf['yc'].values
    ynd = ydf['yn'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    ycf = interp1d(td, ycd)
    ynf = interp1d(td, ynd)
    yc = np.array([ycf(ti) for ti in t])
    yn = np.array([ynf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    y = np.column_stack([yc, yn])
    return t, y, u


def fit_itranslo(t, y, u, results_dir):
    y0 = y[0, :]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_itranslo(t, y0, uf, params)
        res = np.mean(np.square(ym - y))
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
    dyc = np.median(np.absolute(np.ediff1d(y[:, 0])))
    dyn = np.median(np.absolute(np.ediff1d(y[:, 1])))
    a0 = dyn/dyc
    kmax = 10**np.floor(np.log10(np.max(y)))
    dk = kmax/10
    params = lm.Parameters()
    params.add('ku', value=kmax/2, min=0, max=kmax)
    params.add('kf', value=kmax/2, min=0, max=kmax)
    params.add('kr', value=kmax/2, min=0, max=kmax)
    params.add('a', value=a0, min=1, max=10)
    ta = time.time()
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-1
    )
    print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    # opt_params = dict([('ku', 0.0008942295681229174), ('kf', 0.0005048121271898231), ('kr', 0.0006244090506147587), ('a', 4.632731164948481)])
    with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_itranslo(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t, y[:, 0], color='#BBDEFB', label='Cytoplasm (Data)')
        ax1.plot(tm, ym[:, 0], color='#1976D2', label='Cytoplasm (Model)')
        ax1.plot(t, y[:, 1], color='#ffcdd2', label='Nucleus (Data)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='Nucleus (Model)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def prep_idimer_data(y_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    udf = pd.read_csv(u_csv)
    td = np.around(ydf['t'].values, 1)
    yad = ydf['ya'].values
    ydd = ydf['yd'].values
    t = np.around(np.arange(td[0], td[-1], 0.1), 1)
    yaf = interp1d(td, yad)
    ydf = interp1d(td, ydd)
    ya = np.array([yaf(ti) for ti in t])
    yd = np.array([ydf(ti) for ti in t])
    uta = np.around(udf['ta'].values, 1)
    utb = np.around(udf['tb'].values, 1)
    u = np.zeros_like(t)
    for ta, tb in zip(uta, utb):
        ia = list(t).index(ta)
        ib = list(t).index(tb)
        u[ia:ib] = 1
    y = np.column_stack([ya, ya, yd])
    return t, y, u


def fit_idimer(t, y, u, results_dir):
    # ya = (-y + 2*np.min(y) + (np.max(y)-np.min(y)))
    # y0 = [ya[0], ya[0], y[0]]
    y0 = [0.5, 0.5, y[0]]
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    def residual(params):
        tm, ym = sim_idimer(t, y0, uf, params)
        res = (ym[:, 2] - y)
        # res[:600] *= 10
        res = np.sum(res**2)
        # res = np.sum((ym[:, 2] - y[:, 2])**2)
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
    # ta = time.time()
    # params = lm.Parameters()
    # params.add('kf', min=0, max=0.1, brute_step=0.01)
    # params.add('kr', min=0, max=0.1, brute_step=0.01)
    # params.add('ku', min=0, max=1, brute_step=0.1)
    # results = lm.minimize(
    #     residual, params, method='brute',
    #     iter_cb=opt_iter, nan_policy='propagate',
    # )
    # params0 = results.params.valuesdict()
    # params = lm.Parameters()
    # params.add('kf', value=params0['kf'], min=0, max=0.1)
    # params.add('kr', value=params0['kr'], min=0, max=0.1)
    # params.add('ku', value=params0['ku'], min=0, max=1)
    # ta = time.time()
    params = lm.Parameters()
    params.add('kf', value=0.0, min=0.0, max=1.0)
    params.add('kr', value=0.0, min=0.0, max=1.0)
    params.add('ku', value=0.0, min=0.0, max=1.0)
    results = lm.minimize(
        residual, params, method='differential_evolution',
        iter_cb=opt_iter, nan_policy='propagate', tol=1e-9
    )
    # # print('Elapsed Time: ', str(time.time() - ta))
    opt_params = results.params.valuesdict()
    # opt_params = dict([('kf', 0.00015374365541664936), ('kr', 0.003233611739077602), ('ku', 0.03129990345578715)])
    with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    tm, ym = sim_idimer(t, y0, uf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(t, y, color='#ffcdd2', label='AB (Data)')
        ax1.plot(tm, ym[:, 2], color='#d32f2f', label='AB (Model)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


def prep_iexpress_data(y_csv, x_csv, u_csv):
    ydf = pd.read_csv(y_csv)
    xdf = pd.read_csv(x_csv)
    udf = pd.read_csv(u_csv)
    t = np.around(ydf['t'].values/60)
    t = np.arange(t[0], t[-1])
    y = ydf['y'].values[:-1]
    x = xdf['yn'].values[::60]
    uta = np.around(udf['ta'].values/60)
    utb = np.around(udf['tb'].values/60)
    u = np.zeros_like(t)
    ia = list(t).index(uta[0])
    ib = list(t).index(utb[-1])
    u[ia:ib] = 1
    # plt.plot(t, u)
    # plt.plot(t, x)
    # plt.plot(t, y)
    # plt.show()
    return t, y, x, u


def fit_iexpress(t, y, x, u, results_dir):
    y0 = [0, y[0]]
    xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    min_res = 1e15
    best_params = None
    iter_t = time.time()
    # def residual(params):
    #     tm, ym = sim_iexpress(t, y0, xf, params)
    #     res = np.mean((ym[:, 1] - y)**2)
    #     return  res
    # def opt_iter(params, iter, res):
    #     nonlocal min_res, best_params, iter_t
    #     clear_screen()
    #     ti = time.time()
    #     print('seconds/iter:', str(ti - iter_t))
    #     iter_t = ti
    #     print('Iter: {} | Res: {}'.format(iter, res))
    #     print(params.valuesdict())
    #     if res < min_res:
    #         min_res = res
    #         best_params = params.valuesdict()
    #     print('Best so far:')
    #     print('Res:', str(min_res))
    #     print(best_params)
    # ta = time.time()
    # # params = lm.Parameters()
    # # params.add('ka', min=0.01, max=1, brute_step=0.01)
    # # params.add('kb', min=0.1, max=10, brute_step=0.1)
    # # params.add('kc', min=0.001, max=0.1, brute_step=0.001)
    # # params.add('n', min=1, vary=False)
    # # results = lm.minimize(
    # #     residual, params, method='brute',
    # #     iter_cb=opt_iter, nan_policy='propagate',
    # # )
    # params = lm.Parameters()
    # params.add('ka', value=0.1, min=0, max=1)
    # params.add('kb', value=0.1, min=0, max=1)
    # params.add('kc', value=0.1, min=0, max=1)
    # params.add('n', value=1, min=0, max=1)
    # params.add('kf', value=0.1, min=0, max=1)
    # params.add('kg', value=0.1, min=0, max=1)
    # ta = time.time()
    # results = lm.minimize(
    #     residual, params, method='dual_annealing',
    #     iter_cb=opt_iter, nan_policy='propagate', tol=1e-3
    # )
    # print('Elapsed Time: ', str(time.time() - ta))
    # opt_params = params
    # opt_params = results.params.valuesdict()
    # with open(os.path.join(results_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
    #     json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    opt_params = dict([('ka', 0.10014001562994905), ('kb', 0.17638515989679424), ('kc', 0.002468692274783906), ('n', 0.9758946268030773), ('kf', 0.0008406127261191349), ('kg', 0.6462795804896208)])
    tm, ym = sim_iexpress(t, y0, xf, opt_params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.scatter(t, y, color='#ffcdd2', label='POI (Data)')
        # ax1.plot(tm, ym[:, 0], color='#2196F3', label='mRNA (Model)')
        ax1.plot(tm, ym[:, 1], color='#d32f2f', label='POI (Model)')
        ax1.set_xlabel('Time (m)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        fig.savefig(os.path.join(results_dir, 'fit.png'),
            dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)
    return opt_params


if __name__ == '__main__':
    # y_csv = '/home/phuong/data/LINTAD/LINuS-results/0/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LINuS-results/0/'
    # t, y, u = prep_itranslo_data(y_csv, u_csv)
    # fit_itranslo(t, y, u, res_dir)


    # root_dir = '/home/phuong/data/LINTAD/LINuS-mock/'
    # u_csv = '/home/phuong/data/LINTAD/LINuS/u0.csv'
    # for data_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     y_csv = os.path.join(root_dir, data_dirname, 'y.csv')
    #     res_dir = os.path.join(root_dir, data_dirname)
    #     t, y, u = prep_itranslo_data(y_csv, u_csv)
    #     fit_itranslo(t, y, u, res_dir)
    # ku_vals = []
    # kf_vals = []
    # kr_vals = []
    # a_vals = []
    # for data_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     params_path = os.path.join(root_dir, data_dirname, 'opt_params.json')
    #     with open(params_path) as f:
    #         params = json.load(f)
    #         ku_vals.append(params['ku'])
    #         kf_vals.append(params['kf'])
    #         kr_vals.append(params['kr'])
    #         a_vals.append(params['a'])
    # data = [ku_vals, kf_vals, kr_vals]
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [6, 2]})
    #     ax0 = sns.violinplot(data=data, ax=ax0, palette=['#1976D2', '#D32F2F', '#388E3C'])
    #     ax0.set_xticklabels(['ku', 'kf', 'kr'])
    #     ax1 = sns.violinplot(data=a_vals, ax=ax1, color='#F57C00')
    #     ax1.set_xticklabels(['a'])
    #     fig.savefig(os.path.join(root_dir, 'params_dist.png'),
    #         dpi=200, bbox_inches='tight', transparent=False, pad_inches=0)
    #     plt.close(fig)


    # root_dir = '/home/phuong/data/ILID/RA-16I/20200921-B3-sspBu_RA-16I_spike/results/'
    # u_csv = os.path.join(root_dir, 'u_combined.csv')
    # y_csv = os.path.join(root_dir, 'y_combined.csv')
    # u_data = pd.read_csv(u_csv)
    # tu = u_data['tu_ave'].values
    # u = u_data['u_ave'].values
    # uf = interp1d(tu, u, bounds_error=False, fill_value=0)
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y_ave'].values
    # # y = y - y[0]
    # yf = interp1d(t, y, fill_value='extrapolate')
    # y = np.array([yf(ti) for ti in tu])
    # fit_ilid(tu, y, u, root_dir)

    # t = np.arange(0, 600)
    # y0 = [y[0], 0]
    # u = np.zeros_like(t)
    # # p = 20
    # # w = 1
    # # for i in range(t[60], t[480], p):
    # #     u[i:i+w] = 1
    # u[60:480] = 1
    # params = {
    #     "ka": 1.029336557988737,
    #     "kb": 1.2472229538949358,
    #     "kc": 0.34861647650637984,
    #     "n": 1.0961954940397614
    # }
    # uf = interp1d(t, u, bounds_error=False, fill_value=0)
    # tm, ym = sim_ilid(t, y0, uf, params)
    # with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
    #     fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
    #     ax0.plot(t, u)
    #     ax0.set_yticks([0, 1])
    #     ax0.set_ylabel('BL')
    #     ax1.plot(tm, ym[:, 0], color='#1976D2', label='A')
    #     ax1.plot(tm, ym[:, 1], color='#d32f2f', label='B')
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('AU')
    #     ax1.legend(loc='best')
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     save_dir = '/home/phuong/data/ILID/RA-HF/20200921-B3-sspBu_RA-27V_spike/results/'
    #     fig.savefig(os.path.join(save_dir, 'sim.png'),
    #         dpi=300, bbox_inches='tight', transparent=False)
    #     plt.close(fig)


    y0 = [0.1, 0, 0.5, 0, 0.05]
    # uf = interp1d(tu, u, bounds_error=False, fill_value=0)
    t = np.arange(0, 600)
    u = np.zeros_like(t)
    p = 20
    w = 1
    for i in range(t[60], t[540], p):
        u[i:i+w] = 1
    uf = interp1d(t, u, bounds_error=False, fill_value=0)
    params = {
        'k1f': 0.23593737155206962,
        'k1r': 0.005057247900003281,
        'k2f': 0.5858908062641166,
        'k2r': 0.3465577497760164,
    }
    tm, ym = sim_fresca(t, y0, uf, params)
    with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 8]})
        ax0.plot(t, u)
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('BL')
        ax1.plot(tm, ym[:, 1], color='#1976D2', label='iLID_slow')
        ax1.plot(tm, ym[:, 3], color='#d32f2f', label='iLID_fast')
        ax1.plot(tm, ym[:, 4], color='#388E3C', label='sspB')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('AU')
        ax1.legend(loc='best')
        fig.tight_layout()
        fig.canvas.draw()
        save_dir = '/home/phuong/data/ILID/FResCA/'
        fig.savefig(os.path.join(save_dir, 'sim1.png'),
            dpi=300, bbox_inches='tight', transparent=False)
        plt.close(fig)


    # y_csv = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/y.csv'
    # u_csv = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/u.csv'
    # res_dir = '/home/phuong/data/ILID/RA-HF/20200804-RA-HF/results/5/'
    # u_data = pd.read_csv(u_csv)
    # tu = u_data['t'].values
    # u = u_data['u'].values
    # y_data = pd.read_csv(y_csv)
    # t = y_data['t'].values
    # y = y_data['y'].values
    # yf = interp1d(t, y, fill_value='extrapolate')
    # y = np.array([yf(ti) for ti in tu])
    # # ya = (-y + 2*np.min(y) + (np.max(y)-np.min(y)))
    # fit_CaM_M13(tu, y, u, res_dir)


    # y_csv = '/home/phuong/data/LINTAD/LexA-results/0/y.csv'
    # x_csv = '/home/phuong/data/LINTAD/TF/y.csv'
    # u_csv = '/home/phuong/data/LINTAD/LexA/u0.csv'
    # res_dir = '/home/phuong/data/LINTAD/LexA-results/0/'
    # t, y, x, u = prep_iexpress_data(y_csv, x_csv, u_csv)
    # x = np.zeros_like(t)
    # x[130:] = 1
    # fit_iexpress(t, y, x, u, res_dir)
    
    # xf = interp1d(t, x, bounds_error=False, fill_value='extrapolate')
    # y0 = [0, y[0]]
    # params = {'ka': 0.122, 'kb': 0.1, 'kc': 0.2, 'n': 1, 'kf': 0.4, 'kg': 0.1}
    # ta = time.time()
    # tm, ym = sim_iexpress(t, y0, xf, params)
    # print(time.time() - ta)
    # plt.plot(t, y)
    # plt.plot(tm, ym[:, 0])
    # plt.show()

    # t = np.arange(0, 100, 1)
    # omega = 20
    # tau = 32
    # n = 4
    # C = []
    # for ti in t:
    #     if ti >= 32 and ti < 35:
    #         Ci = 0.1 + 0.9*np.sin(omega*(ti - tau))**n
    #     else:
    #         Ci = 0.1
    #     C.append(Ci)
    # # plt.plot(t, C)
    # # plt.show()
    # C0 = np.ones_like(t) * 0.1
    # Cf = interp1d(t, C0, bounds_error=False, fill_value=0.1)
    # y0 = [0, 0, 0, 0, 0]
    # t, y = sim_CaM_M13(t, y0, Cf)
    # y0 = y[-1, :]
    # Cf = interp1d(t, C, bounds_error=False, fill_value=0.1)
    # t, y = sim_CaM_M13(t, y0, Cf)
    # Pb = y[:, 2] + y[:, 3] + y[:, 4]
    # # plt.plot(t, C)
    # plt.plot(t, y[:, 3])
    # plt.show()