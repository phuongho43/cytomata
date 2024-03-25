import os
import sys
import time
import shutil
import ast

sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from natsort import natsorted

from cytomata.model import sim_lov, sim_ilid, sim_signet, calc_hypervolume2D, NSGAII
from cytomata.plot import plot_uy, plot_groups
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def obj_func_opto_ppi(individual):
    """Objective function for fitting the LOV or iLID model to ddFP data.

    Maximize the negative mean squared error between biosensor data and model response.

    Args:
        individual (1 x 3 array): [kl, kd, kb] parameters

    Returns:
        obj_scores (1 x 1 array): objective score to be maximized (-MSE)
    """
    tm, Xm = ode_model(tu_data, y0_data, uu_data, individual)
    yy_model = Xm[-1]
    # t_idx: the corresponding time indices between data vs model sim
    # for us, time interval for data is 5s but model sim interval is 0.1s (due to tu_data)
    obj = -np.mean((yy_data - yy_model[t_idx])**2)
    return np.array([obj])


if __name__ == '__main__':
    results_dps = [
        '/home/phuong/protosignet/1--biosensor/1--training/1--LOV/0--I427V/results/',
        '/home/phuong/protosignet/1--biosensor/1--training/1--LOV/1--V416I/results/',
        '/home/phuong/protosignet/1--biosensor/1--training/2--iLID/0--I427V/results/',
        '/home/phuong/protosignet/1--biosensor/1--training/2--iLID/1--V416I/results/',
    ]
    save_dps = [
        '/home/phuong/protosignet/1--biosensor/6--models/0--LOV/0--I427V/',
        '/home/phuong/protosignet/1--biosensor/6--models/0--LOV/1--V416I/',
        '/home/phuong/protosignet/1--biosensor/6--models/1--iLID/0--I427V/',
        '/home/phuong/protosignet/1--biosensor/6--models/1--iLID/1--V416I/',
    ]
    for results_dp, save_dp in zip(results_dps, save_dps):
        # if os.path.exists(save_dp):
        #     input('Directory already exists. Overwrite?...Press ENTER to continue, Ctrl-C to abort.')
        setup_dirs(save_dp)
        shutil.copyfile('fit_model.py', os.path.join(save_dp, 'py_code.txt'))

        u_df = pd.read_csv(os.path.join(results_dp, 'u.csv'))
        y_df = pd.read_csv(os.path.join(results_dp, 'y.csv'))
        u_df = u_df.groupby('t', as_index=False)['u'].mean()
        y_df = y_df.groupby('t', as_index=False)['y'].mean()
        tu_data = u_df['t'].values
        uu_data = u_df['u'].values
        ty_data = y_df['t'].values
        yy_data = y_df['y'].values
        if 'LOV' in results_dp:
            ode_model = sim_lov
            y0_data = [0, 0, 0, -np.min(yy_data)]
            yy_data = yy_data - np.min(yy_data)
        else:
            ode_model = sim_ilid
            y0_data = [np.max(yy_data), 0, np.max(yy_data), 0]
        # Biosensor timelapse has interval of 5s but stimuli has interval of 0.1s
        # Model simulated using stimuli timepoints also has interval of 0.1s
        # MSE calculation --> use only the matching timepoints
        t_idx = np.isin(tu_data, ty_data)

        param_space = [
            list(np.arange(0.001, 100.001, 0.001)),  # kl
            list(np.arange(0.001, 100.001, 0.001)),  # kd
            list(np.arange(0.001, 100.001, 0.001)),  # kb
        ]

        for run_i in range(5):
            print(save_dp, run_i)
            opto = NSGAII(
                obj_func=obj_func_opto_ppi,
                param_space=param_space,
                pop_size=200,
                rng_seed=None
            )
            data = opto.evolve(
                rec_rate=0.1, mut_rate=0.3, mut_spread=0.01, n_gen=250,
            )
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(save_dp, f'{run_i}.csv'), index=False)


    print('\a')
