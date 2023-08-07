import os
import sys
import time
import json
import random
import warnings
sys.path.append(os.path.abspath('../'))


import numpy as np
import lmfit as lm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, argrelextrema
from natsort import natsorted
from tabulate import tabulate
import pygad

from cytomata.process import process_fit_data, rescale_01
from cytomata.model import sim_ilid, sim_lov, sim_sparser, gen_model, sim_model, sim_damped_osc
from cytomata.plot import plot_class_group, plot_uy, plot_netw_fitn
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette



if __name__ == '__main__':
    save_dir = '/home/phuong/data/1-fakr/0-model/test/'
    plot_netw_fitn(save_dir=save_dir, n_fit=[0, 1], y_fit=[4, 12], t_conc=[1, 2, 3], y_conc=[2, 4, 8], params=np.ones(15))
