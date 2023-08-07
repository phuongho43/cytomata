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
from tabulate import tabulate

from cytomata.process import process_fit_data
from cytomata.model import sim_ilid, sim_lov, sim_sparser
from cytomata.plot import plot_class_group, plot_uy
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def fit_ddfp_dimer_model(root_dir, model_type):
    fast_dir = os.path.join(root_dir, 'I427V', 'results')
    fast_td, fast_yd, fast_sem, fast_tu, fast_ud, fast_uf = process_fit_data(fast_dir)
    slow_dir = os.path.join(root_dir, 'V416I', 'results')
    slow_td, slow_yd, slow_sem, slow_tu, slow_ud, slow_uf = process_fit_data(slow_dir)
    if model_type == 'lov':
        model_func = sim_lov
        fast_peak_idx = np.argmin(fast_yd)
        slow_peak_idx = np.argmin(slow_yd)
        fast_y0 = [0, 0, 0, -np.min(fast_yd)]
        slow_y0 = [0, 0, 0, -np.min(slow_yd)]
        fast_yd = fast_yd - np.min(fast_yd)
        slow_yd = slow_yd - np.min(slow_yd)
    elif model_type == 'ilid':
        model_func = sim_ilid
        fast_peak_idx = np.argmax(fast_yd)
        slow_peak_idx = np.argmax(slow_yd)
        fast_y0 = [np.max(fast_yd), 0, np.max(fast_yd), 0]
        slow_y0 = [np.max(slow_yd), 0, np.max(slow_yd), 0]
    else:
        raise ValueError('Unknown model type {}'.format(model_type))
    params = lm.Parameters()
    params.add('kl', value=0.0, min=0.0, max=100.0)
    params.add('kd1', value=0.0, min=0.0, max=10.0)
    params.add('kd2', value=0.0, min=0.0, max=10.0)
    params.add('kb', value=0.0, min=0.0, max=100.0)
    min_res = np.inf
    best_params = {}
    ta = time.time()
    iter_ti = time.time()
    iter_vals = []
    res_vals = []
    def residual(params):
        params = params.valuesdict()
        fast_params = {'kl': params['kl'], 'kd': params['kd1'], 'kb': params['kb']}
        slow_params = {'kl': params['kl'], 'kd': params['kd2'], 'kb': params['kb']}
        fast_tm, fast_ym = model_func(fast_td, fast_y0, fast_uf, fast_params)
        slow_tm, slow_ym = model_func(slow_td, slow_y0, slow_uf, slow_params)
        fast_ym = fast_ym[-1]
        slow_ym = slow_ym[-1]
        fast_err = ((fast_ym - fast_yd)**2)/(fast_sem)
        slow_err = ((slow_ym - slow_yd)**2)/(slow_sem)
        fast_err[fast_peak_idx] *= 10
        slow_err[slow_peak_idx] *= 10
        comb_err = np.concatenate((fast_err, slow_err), axis=None)
        return comb_err
    def opt_iter(params_i, iter_i, res_i):
        nonlocal min_res, best_params, iter_ti, iter_vals , res_vals
        ti = time.time()
        iter_dt = ti - iter_ti
        iter_ti = time.time()
        params_i = params_i.valuesdict()
        res_i = np.mean(res_i)
        if res_i < min_res:
            min_res = res_i
            best_params = params_i
        iter_vals.append(iter_i)
        res_vals.append(min_res)
        main_tabu = [
            ['tot_time', time.time() - ta],
            ['seconds/iter', iter_dt],
            ['iter_n', iter_i],
            ['min_res', min_res]]
        params_tabu = [[k, v] for k, v in best_params.items()]
        print(tabulate(main_tabu + params_tabu))
    results = lm.minimize(
        residual, params, method='nelder', tol=1e-6,
        iter_cb=opt_iter, nan_policy='omit', calc_covar=True, reduce_fcn=np.mean)
    opt_params = results.params.valuesdict()
    with open(os.path.join(root_dir, 'opt_params.json'), 'w', encoding='utf-8') as fo:
        json.dump(opt_params, fo, ensure_ascii=False, indent=4)
    with open(os.path.join(root_dir, 'fit_report.txt'), 'w', encoding='utf-8') as f:
        f.write(lm.fit_report(results))
    res_df = pd.DataFrame({'t': iter_vals, 'y': res_vals})
    res_df.to_csv(os.path.join(root_dir, 'res.csv'), index=False)
    return opt_params


def plot_figure_1():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/'
    results_dir = os.path.join(root_dir, 'I427V', 'results')
    td1, yd1, sem1, tu, ud, uf = process_fit_data(results_dir)
    y01 = [0, 0, 0, -np.min(yd1)]
    yd1 = yd1 - np.min(yd1)
    results_dir = os.path.join(root_dir, 'V416I', 'results')
    td2, yd2, sem2, tu, ud, uf = process_fit_data(results_dir)
    y02 = [0, 0, 0, -np.min(yd2)]
    yd2 = yd2 - np.min(yd2)
    with open(os.path.join(root_dir, 'opt_params.json')) as f:
        params = json.load(f)
    params1 = {'kl': params['kl'], 'kd': params['kd1'], 'kb': params['kb']}
    tm1, ym1 = sim_lov(td1, y01, uf, params1)
    ym1 = ym1[-1]
    params2 = {'kl': params['kl'], 'kd': params['kd2'], 'kb': params['kb']}
    tm2, ym2 = sim_lov(td2, y02, uf, params2)
    ym2 = ym2[-1]
    yd1_df = pd.DataFrame({'t': td1, 'y': yd1, 'h': np.ones_like(td1)*0})
    yd2_df = pd.DataFrame({'t': td2, 'y': yd2, 'h': np.ones_like(td2)*1})
    ym1_df = pd.DataFrame({'t': tm1, 'y': ym1, 'h': np.ones_like(tm1)*0})
    ym2_df = pd.DataFrame({'t': tm2, 'y': ym2, 'h': np.ones_like(tm2)*1})
    yd_df = pd.concat([yd1_df, yd2_df], ignore_index=True)
    ym_df = pd.concat([ym1_df, ym2_df], ignore_index=True)
    u_df = pd.DataFrame({'t': tu, 'u': ud})
    palette = ['#785EF0', '#FE6100']
    group_labels = ['LOVfast', 'LOVslow']
    plot_uy(ym_df, u_df, root_dir, yd_df=yd_df, fname='lov-fit.png', overlay_tu=False, dpi=300, lgd_loc='lower right',
        ylabel='AU', ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_2():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/'
    results_dir = os.path.join(root_dir, 'I427V', 'results')
    td1, yd1, sem1, tu, ud, uf = process_fit_data(results_dir)
    y01 = [np.max(yd1), 0, np.max(yd1), 0]
    results_dir = os.path.join(root_dir, 'V416I', 'results')
    td2, yd2, sem2, tu, ud, uf = process_fit_data(results_dir)
    y02 = [np.max(yd2), 0, np.max(yd2), 0]
    with open(os.path.join(root_dir, 'opt_params.json')) as f:
        params = json.load(f)
    params1 = {'kl': params['kl'], 'kd': params['kd1'], 'kb': params['kb']}
    tm1, ym1 = sim_ilid(td1, y01, uf, params1)
    ym1 = ym1[-1]
    params2 = {'kl': params['kl'], 'kd': params['kd2'], 'kb': params['kb']}
    tm2, ym2 = sim_ilid(td2, y02, uf, params2)
    ym2 = ym2[-1]
    yd1_df = pd.DataFrame({'t': td1, 'y': yd1, 'h': np.ones_like(td1)*0})
    yd2_df = pd.DataFrame({'t': td2, 'y': yd2, 'h': np.ones_like(td2)*1})
    ym1_df = pd.DataFrame({'t': tm1, 'y': ym1, 'h': np.ones_like(tm1)*0})
    ym2_df = pd.DataFrame({'t': tm2, 'y': ym2, 'h': np.ones_like(tm2)*1})
    yd_df = pd.concat([yd1_df, yd2_df], ignore_index=True)
    ym_df = pd.concat([ym1_df, ym2_df], ignore_index=True)
    u_df = pd.DataFrame({'t': tu, 'u': ud})
    palette = ['#785EF0', '#FE6100']
    group_labels = ['iLIDfast', 'iLIDslow']
    plot_uy(ym_df, u_df, root_dir, yd_df=yd_df, fname='ilid-fit.png', overlay_tu=False, dpi=300, lgd_loc='upper right',
        ylabel='AU', ulabel='BL', group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_3():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    tm, ym = sim_sparser(td, y0, uf, params)
    ym = ym[-1]
    yd_df = pd.DataFrame({'t': td, 'y': yd, 'h': np.zeros_like(td)})
    ym_df = pd.DataFrame({'t': tm, 'y': ym, 'h': np.ones_like(tm)})
    yx_df = pd.concat([yd_df, ym_df], ignore_index=True)
    u_df = pd.DataFrame({'t': tu, 'u': ud})
    palette = [(33/256, 33/256, 33/256, 0.5), '#DC267F']
    group_labels = ['Data', 'Model']
    plot_uy(yx_df, u_df, root_dir, yd_df=yd_df, fname='composite.png', dpi=300, overlay_tu=True,
        lgd_loc='best', ylabel='AU', ulabel='BL', group_labels=group_labels,
        ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_4():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/'
    lov_de_csv = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/res.csv'
    lov_ne_csv = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/1-nelder/res.csv'
    lov_de_df = pd.read_csv(lov_de_csv)
    lov_ne_df = pd.read_csv(lov_ne_csv)
    lov_de_df['h'] = np.ones_like(lov_de_df['t'])*0
    lov_ne_df['h'] = np.ones_like(lov_ne_df['t'])*1
    y_df = pd.concat([lov_de_df, lov_ne_df], ignore_index=True)
    palette = ['#785EF0', '#FE6100']
    group_labels = ['Differential Evolution', 'Nelder-Mead']
    plot_uy(y_df, u_df=None, save_dir=root_dir, fname='res.png', dpi=300, lgd_loc='best',
        ylabel='Residual (MSE)', xlabel='Iteration', group_labels=group_labels,
        ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_5():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    tm, ym = sim_sparser(td, y0, uf, params)
    ym1 = ym[5] + ym[6] + ym[8]
    ym2 = ym[7] + ym[8]
    ym1_df = pd.DataFrame({'t': tm, 'y': ym1, 'h': np.ones_like(tm)*0})
    ym2_df = pd.DataFrame({'t': tm, 'y': ym2, 'h': np.ones_like(tm)*1})
    yx_df = pd.concat([ym1_df, ym2_df], ignore_index=True)
    u_df = pd.DataFrame({'t': tu, 'u': ud})
    palette = ['#FE6100', '#785EF0']
    group_labels = ['Bound LOVfast', 'Bound iLIDslow']
    plot_uy(yx_df, u_df, root_dir, fname='AB-BC.png', dpi=300, overlay_tu=True, lgd_loc='lower right',
        ylabel='AU', ulabel='BL', group_labels=group_labels,
        ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_6():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    AUCs = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_sparser(tu, y0, uf, params)
    #     ym = ym[-1]
    #     AUCs.append(simps(ym, tm))
    # AUCs = np.array(AUCs)
    # y_df = pd.DataFrame({'t': freqs, 'y': AUCs})
    # y_df.to_csv(os.path.join(root_dir, 'AUC_sparse.csv'), index=False)
    y_df = pd.read_csv(os.path.join(root_dir, 'AUC_sparse.csv'))
    palette = ['#DC267F']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_sparse.png', dpi=300, lgd_loc='best',
        ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)', group_labels=None,
        logx=logx, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_7():
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    results_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(results_dir)
    y0 = [np.max(yd), 0, np.max(yd), 0]
    with open(os.path.join(root_dir, 'opt_params.json')) as f:
        params = json.load(f)
    params = {'kl': params['kl'], 'kd': params['kd1'], 'kb': params['kb']}
    AUCs = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_ilid(tu, y0, uf, params)
    #     ym = ym[-1]
    #     AUCs.append(simps(ym, tm))
    # AUCs = np.array(AUCs)
    # y_df = pd.DataFrame({'t': freqs, 'y': AUCs})
    # y_df.to_csv(os.path.join(save_dir, 'AUC_dense.csv'), index=False)
    y_df = pd.read_csv(os.path.join(save_dir, 'AUC_dense.csv'))
    palette = ['#785EF0']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_dense.png', dpi=300, lgd_loc='best',
        ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)', group_labels=None,
        logx=logx, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_8():
    rescale01 = lambda y: (y - min(y))/(max(y) - min(y))
    ddfp_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    gexp_dir = '/home/phuong/data/1-fakr/2-transcription/4--dense-sparse/results/'
    ddfp_dense_df = pd.read_csv(os.path.join(ddfp_dir, 'AUC_dense.csv'))
    ddfp_dense_df['y'] = rescale01(ddfp_dense_df['y'])
    ddfp_dense_df['h'] = np.ones_like(ddfp_dense_df['y'])*0
    gexp_dense_df = pd.read_csv(os.path.join(gexp_dir, 'y.csv'))
    gexp_dense_df = gexp_dense_df.loc[(gexp_dense_df['class'] == 0), ['group', 'response', 'repeat']]
    gexp_dense_df = gexp_dense_df.replace({'group': {0: 1/300, 1: 0.05, 2: 0.1, 3: 0.25, 4: 0.5, 5: 1}})
    gexp_dense_df.columns = ['t', 'y', 'h']
    for h in gexp_dense_df['h'].unique():
        gexp_dense_df.loc[(gexp_dense_df['h'] == h), 'y'] = rescale01(gexp_dense_df.loc[(gexp_dense_df['h'] == h), 'y'])
    gexp_dense_df['h'] = np.ones_like(gexp_dense_df['y'])*1
    y_df = pd.concat([ddfp_dense_df, gexp_dense_df])
    palette = ['#785EF0', '#d7cffb']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    group_labels = ['ddFP Model', 'Transcription Data']
    plot_uy(y_df, u_df=None, save_dir=ddfp_dir, fname='dense_ddfp_vs_gexp.png', dpi=300,
        logx=logx, lgd_loc='upper left', ylabel='Normalized Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_9():
    rescale01 = lambda y: (y - min(y))/(max(y) - min(y))
    ddfp_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    gexp_dir = '/home/phuong/data/1-fakr/2-transcription/4--dense-sparse/results/'
    ddfp_sparse_df = pd.read_csv(os.path.join(ddfp_dir, 'AUC_sparse.csv'))
    ddfp_sparse_df['y'] = rescale01(ddfp_sparse_df['y'])
    ddfp_sparse_df['h'] = np.ones_like(ddfp_sparse_df['y'])*0
    gexp_sparse_df = pd.read_csv(os.path.join(gexp_dir, 'y.csv'))
    gexp_sparse_df = gexp_sparse_df.loc[(gexp_sparse_df['class'] == 1), ['group', 'response', 'repeat']]
    gexp_sparse_df = gexp_sparse_df.replace({'group': {0: 1/300, 1: 0.05, 2: 0.1, 3: 0.25, 4: 0.5, 5: 1}})
    gexp_sparse_df.columns = ['t', 'y', 'h']
    for h in gexp_sparse_df['h'].unique():
        gexp_sparse_df.loc[(gexp_sparse_df['h'] == h), 'y'] = rescale01(gexp_sparse_df.loc[(gexp_sparse_df['h'] == h), 'y'])
    gexp_sparse_df['h'] = np.ones_like(gexp_sparse_df['y'])*1
    y_df = pd.concat([ddfp_sparse_df, gexp_sparse_df])
    palette = ['#DC267F', '#f5bed9']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    group_labels = ['ddFP Model', 'Transcription Data']
    plot_uy(y_df, u_df=None, save_dir=ddfp_dir, fname='sparse_ddfp_vs_gexp.png', dpi=300,
        logx=logx, lgd_loc='lower left', ylabel='Normalized Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_10():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    baseline_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    faster_lov_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1']*10,
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    slower_lov_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1']*0.1,
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    AUCs_baseline = []
    AUCs_faster_lov = []
    AUCs_slower_lov = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_sparser(tu, y0, uf, baseline_params)
    #     AUCs_baseline.append(simps(ym[-1], tm))
    #     tm, ym = sim_sparser(tu, y0, uf, faster_lov_params)
    #     AUCs_faster_lov.append(simps(ym[-1], tm))
    #     tm, ym = sim_sparser(tu, y0, uf, slower_lov_params)
    #     AUCs_slower_lov.append(simps(ym[-1], tm))
    # AUCs_baseline = np.array(AUCs_baseline)
    # AUCs_faster_lov = np.array(AUCs_faster_lov)
    # AUCs_slower_lov = np.array(AUCs_slower_lov)
    # y0_df = pd.DataFrame({'t': freqs, 'y': AUCs_faster_lov, 'h': np.ones_like(freqs)*0})
    # y1_df = pd.DataFrame({'t': freqs, 'y': AUCs_baseline, 'h': np.ones_like(freqs)*1})
    # y2_df = pd.DataFrame({'t': freqs, 'y': AUCs_slower_lov, 'h': np.ones_like(freqs)*2})
    # y_df = pd.concat([y0_df, y1_df, y2_df], ignore_index=True)
    # y_df.to_csv(os.path.join(root_dir, 'AUC_vary_lov.csv'), index=False)
    y_df = pd.read_csv(os.path.join(root_dir, 'AUC_vary_lov.csv'))
    group_labels = ['faster LOV reversion', 'base LOV reversion', 'slower LOV reversion']
    palette = ['#420b26', '#DC267F', '#f5bed9']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_vary_lov.png', dpi=300, lgd_loc='lower left',
        logx=logx, ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_11():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    baseline_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    faster_ilid_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2']*10,
        "kb2": ilid_params['kb'],
    }
    slower_ilid_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2']*0.1,
        "kb2": ilid_params['kb'],
    }
    AUCs_baseline = []
    AUCs_faster_ilid = []
    AUCs_slower_ilid = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_sparser(tu, y0, uf, baseline_params)
    #     AUCs_baseline.append(simps(ym[-1], tm))
    #     tm, ym = sim_sparser(tu, y0, uf, faster_ilid_params)
    #     AUCs_faster_ilid.append(simps(ym[-1], tm))
    #     tm, ym = sim_sparser(tu, y0, uf, slower_ilid_params)
    #     AUCs_slower_ilid.append(simps(ym[-1], tm))
    # AUCs_baseline = np.array(AUCs_baseline)
    # AUCs_faster_ilid = np.array(AUCs_faster_ilid)
    # AUCs_slower_ilid = np.array(AUCs_slower_ilid)
    # y0_df = pd.DataFrame({'t': freqs, 'y': AUCs_faster_ilid, 'h': np.ones_like(freqs)*0})
    # y1_df = pd.DataFrame({'t': freqs, 'y': AUCs_baseline, 'h': np.ones_like(freqs)*1})
    # y2_df = pd.DataFrame({'t': freqs, 'y': AUCs_slower_ilid, 'h': np.ones_like(freqs)*2})
    # y_df = pd.concat([y0_df, y1_df, y2_df], ignore_index=True)
    # y_df.to_csv(os.path.join(root_dir, 'AUC_vary_ilid.csv'), index=False)
    y_df = pd.read_csv(os.path.join(root_dir, 'AUC_vary_ilid.csv'))
    group_labels = ['faster iLID reversion', 'base iLID reversion', 'slower iLID reversion']
    palette = ['#420b26', '#DC267F', '#f5bed9']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_vary_ilid.png', dpi=300, lgd_loc='lower left',
        logx=logx, ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_12():
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(root_dir)
    yd = yd -np.min(yd)
    y0 = [0, 0, 0, 0, np.max(yd), np.max(yd), 0, 0, yd[0]]
    lov_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    with open(os.path.join(lov_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    ilid_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    with open(os.path.join(ilid_dir, 'opt_params.json')) as f:
        ilid_params = json.load(f)
    baseline_params = {
        "kl1": lov_params['kl'],
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    lower_induction_params = {
        "kl1": lov_params['kl']*0.1,
        "kd1": lov_params['kd1'],
        "kb1": lov_params['kb'],
        "kl2": ilid_params['kl'],
        "kd2": ilid_params['kd2'],
        "kb2": ilid_params['kb'],
    }
    AUCs_baseline = []
    AUCs_lower_induction = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_sparser(tu, y0, uf, baseline_params)
    #     AUCs_baseline.append(simps(ym[-1], tm))
    #     tm, ym = sim_sparser(tu, y0, uf, lower_induction_params)
    #     AUCs_lower_induction.append(simps(ym[-1], tm))
    # AUCs_baseline = np.array(AUCs_baseline)
    # AUCs_lower_induction = np.array(AUCs_lower_induction)
    # y0_df = pd.DataFrame({'t': freqs, 'y': AUCs_baseline, 'h': np.ones_like(freqs)*1})
    # y1_df = pd.DataFrame({'t': freqs, 'y': AUCs_lower_induction, 'h': np.ones_like(freqs)*2})
    # y_df = pd.concat([y0_df, y1_df], ignore_index=True)
    # y_df.to_csv(os.path.join(root_dir, 'AUC_vary_sparser_induction.csv'), index=False)
    y_df = pd.read_csv(os.path.join(root_dir, 'AUC_vary_sparser_induction.csv'))
    group_labels = ['base LOV induction', 'lower LOV induction']
    palette = ['#DC267F', '#f5bed9']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_vary_sparser_induction.png', dpi=300, lgd_loc='lower left',
        logx=logx, ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_13():
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/dimer_model/0-devol/'
    results_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/I427V/results/'
    td, yd, sem, tu, ud, uf = process_fit_data(results_dir)
    y0 = [np.max(yd), 0, np.max(yd), 0]
    with open(os.path.join(root_dir, 'opt_params.json')) as f:
        params = json.load(f)
    base_induction_params = {'kl': params['kl'], 'kd': params['kd1'], 'kb': params['kb']}
    lower_induction_params = {'kl': params['kl']*0.01, 'kd': params['kd1'], 'kb': params['kb']}
    AUCs_base_induction = []
    AUCs_lower_induction = []
    tu = np.arange(0, 301, 1)
    periods = list(range(1, 20)) + [24, 28, 32, 36, 40, 50, 60, 80, 100, 200, 300, 301]
    periods = np.array(periods)
    freqs = 1/periods
    # for period in periods:
    #     print(period)
    #     ud = np.zeros_like(tu)
    #     ud[period:301:period] = 1
    #     uf = interp1d(tu, ud, kind='nearest')
    #     tm, ym = sim_ilid(tu, y0, uf, base_induction_params)
    #     AUCs_base_induction.append(simps(ym[-1], tm))
    #     tm, ym = sim_ilid(tu, y0, uf, lower_induction_params)
    #     AUCs_lower_induction.append(simps(ym[-1], tm))
    # AUCs_base_induction = np.array(AUCs_base_induction)
    # AUCs_lower_induction = np.array(AUCs_lower_induction)
    # y0_df = pd.DataFrame({'t': freqs, 'y': AUCs_base_induction, 'h': np.ones_like(freqs)*0})
    # y1_df = pd.DataFrame({'t': freqs, 'y': AUCs_lower_induction, 'h': np.ones_like(freqs)*1})
    # y_df = pd.concat([y0_df, y1_df], ignore_index=True)
    # y_df.to_csv(os.path.join(save_dir, 'AUC_vary_denser_induction.csv'), index=False)
    y_df = pd.read_csv(os.path.join(save_dir, 'AUC_vary_denser_induction.csv'))
    group_labels = ['base iLID induction', 'lower iLID induction']
    palette = ['#785EF0', '#d7cffb']
    logx = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    plot_uy(y_df, u_df=None, save_dir=save_dir, fname='AUC_vary_denser_induction.png', dpi=300, lgd_loc='upper left',
        logx=logx, ylabel='Total Output', xlabel='BL Pulsing Frequency (Hz)',
        group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)


def plot_figure_14():
    data_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/I427V/results/'
    params_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/dimer_model/0-devol/'
    td, yd, sem, tu, ud, uf = process_fit_data(data_dir)
    y0 = [0, 0, 0, -np.min(yd)]
    with open(os.path.join(params_dir, 'opt_params.json')) as f:
        lov_params = json.load(f)
    baseline_params = {
        "kl": lov_params['kl'],
        "kd": lov_params['kd1'],
        "kb": lov_params['kb'],
    }
    lower_induction_params = {
        "kl": lov_params['kl'],
        "kd": lov_params['kd1'],
        "kb": lov_params['kb'],
    }
    tu = np.arange(0, 301, 1)
    period = 4
    ud = np.zeros_like(tu)
    ud[60:241:period] = 1
    uf = interp1d(tu, ud, kind='nearest')
    tm0, ym0 = sim_lov(tu, y0, uf, lower_induction_params)
    ym0 = ym0[-1]
    period = 40
    ud = np.zeros_like(tu)
    ud[60:241:period] = 1
    uf = interp1d(tu, ud, kind='nearest')
    tm1, ym1 = sim_lov(tu, y0, uf, lower_induction_params)
    ym1 = ym1[-1]
    ym1_df = pd.DataFrame({'t': tm0, 'y': ym0, 'h': np.ones_like(tm0)*0})
    ym2_df = pd.DataFrame({'t': tm1, 'y': ym1, 'h': np.ones_like(tm1)*1})
    yx_df = pd.concat([ym2_df, ym1_df], ignore_index=True)
    u_df = pd.DataFrame({'t': tu, 'u': ud})
    palette = ['#785EF0', '#d7cffb']
    group_labels = ['0.25Hz BL', '0.025Hz BL']
    save_dir = '/home/phuong/data/1-fakr/1-ddFP/5-validation/results/'
    plot_uy(yx_df, u_df=None, save_dir=save_dir, fname='platforming_effect.png', dpi=300, overlay_tu=False, lgd_loc='lower right',
        ylabel='AU', ulabel='BL', group_labels=group_labels,
        ymin=None, ymax=None, order=None, palette=palette)


if __name__ == '__main__':
    # root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/1-LOV/'
    # fit_ddfp_dimer_model(root_dir, 'lov')
    # plot_figure_1()

    # root_dir = '/home/phuong/data/1-fakr/1-ddFP/4-training/2-iLID/'
    # fit_ddfp_dimer_model(root_dir, 'ilid')
    # plot_figure_2()

    # plot_figure_3()
    # plot_figure_4()
    # plot_figure_5()
    # plot_figure_6()
    # plot_figure_7()
    plot_figure_8()
    plot_figure_9()
    # plot_figure_10()
    # plot_figure_11()
    # plot_figure_12()
    # plot_figure_13()
    # plot_figure_14()
