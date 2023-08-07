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
from scipy.signal import find_peaks
from natsort import natsorted
from tabulate import tabulate
import pygad

from cytomata.process import process_fit_data, rescale_01
from cytomata.model import sim_ilid, sim_lov, sim_sparser, gen_model, sim_model, sim_damped_osc
from cytomata.plot import plot_class_group, plot_uy, plot_netw_fitn, plot_model
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def evolve_oscillator(save_dir):
    setup_dirs(save_dir)
    sample_dir = os.path.join(save_dir, 'sample')
    setup_dirs(sample_dir)
    td0 = np.arange(0, 301, 1)
    td1 = np.arange(0, 301, 1)
    ud0 = np.zeros_like(td0)
    ud1 = np.zeros_like(td1)
    ud1[150] = 1
    uf0 = interp1d(td0, ud0, kind='nearest')
    uf1 = interp1d(td1, ud1, kind='nearest')
    def fitness_fnc(params, params_idx):
        try:
            model, y0 = gen_model(params)
            tm0, ym0 = sim_model(td0, y0, uf0, model)
            y1 = ym0[:, -1]
            tm1, ym1 = sim_model(td1, y1, uf1, model)
            ym1 = ym1[-1]
            n_peaks0 = len(find_peaks(ym1[:150], prominence=0.001)[0])
            n_valls0 = len(find_peaks(-ym1[:150], prominence=0.001)[0])
            n_peaks1 = len(find_peaks(ym1[150:], prominence=0.02)[0])
            n_valls1 = len(find_peaks(-ym1[150:], prominence=0.02)[0])
            cost_params = (np.count_nonzero(params[6:]))/(len(params[6:]) + 1)
            fitness = (n_peaks1*n_valls1 - n_peaks0*n_valls0)/(cost_params**2)
            fitness = 0 if np.isnan(fitness) else fitness
            return fitness
        except Exception as e:
            print(e)
            return 0
    ta = time.time()
    iter_ti = time.time()
    data = []
    def on_generation(ga_instance):
        nonlocal iter_ti, data
        ti = time.time()
        iter_dt = ti - iter_ti
        iter_ti = time.time()
        ti_ta = ti - ta
        xi_gen = ga_instance.generations_completed
        yi_fit = np.max(ga_instance.last_generation_fitness)
        zi_sol = list(ga_instance.last_generation_elitism[0])
        z_col_names = ['t', 'x', 'y']
        z_col_names += [
            'kra', 'krb', 'krc',
            'kua', 'kub', 'kuc',
            'kta', 'ktb', 'ktc',
            'kba', 'kab', 'kac',
            'kca', 'kcb', 'kbc']
        z_col_vals = [ti_ta, xi_gen, yi_fit] + zi_sol
        data.append(dict(zip(z_col_names, z_col_vals)))
        ##
        df = pd.DataFrame(data)
        model, y0 = gen_model(zi_sol)
        tm0, ym0 = sim_model(td0, y0, uf0, model)
        y1 = ym0[:, -1]
        tm1, ym1 = sim_model(td1, y1, uf1, model)
        ym1 = ym1[-1]
        plot_netw_fitn(save_dir=sample_dir, n_fit=df.x.values, y_fit=df.y.values, t_conc=tm1, y_conc=ym1, params=zi_sol)
        ##
        status = [
            ['tot_time', iter_ti - ta],
            ['seconds/iter', iter_dt],
            ['generations', xi_gen],
            ['fitness', list(ga_instance.last_generation_fitness)],
            ['best_sol', list(ga_instance.last_generation_elitism[0])]
        ]
        print(tabulate(status))
    def crossover_func(parents, offsprings_size, ga_instance):
        offsprings = []
        idx = 0
        option = np.random.choice([0, 1, 2], p=[0.9, 0.05, 0.05])
        while len(offsprings) != offsprings_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            if option == 1:
                offsprings.append(parent1)
            elif option == 2:
                offsprings.append(parent2)
            else:
                node = np.random.choice([0, 1, 2])
                child = parent1.copy()
                child[node::3] = parent2[node::3]
                offsprings.append(child)
            idx += 1
        return np.array(offsprings)
    def mutation_func(offsprings, ga_instance):
        n_genes = offsprings.shape[1]
        n_offspr = offsprings.shape[0]
        idx_genes = np.arange(n_genes)
        idx_offspr = np.arange(n_offspr)
        for ospr_i in idx_offspr:
            # For each offspring, determine which genes to mutate
            mut_op = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            if mut_op == 1: # Randomize all genes
                genes_to_mutate = idx_genes
            elif mut_op == 2: # Randomize all genes affecting a random single species
                node_k = np.random.choice([0, 1, 2])
                genes_to_mutate = idx_genes[node_k::3]
            elif mut_op == 3: # Randomize all genes of a random category
                gcats = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11, 12, 13, 14]}
                genes_to_mutate = gcats[np.random.choice([0, 1, 2, 3])]
            else:
                n_mutations = np.random.choice(idx_genes)
                genes_to_mutate = np.random.choice(idx_genes, size=n_mutations, replace=False)
            # print(genes_to_mutate)
            for gene_i in genes_to_mutate:
                gene_i_space = ga_instance.gene_space[gene_i]
                gene_i_val = offsprings[ospr_i, gene_i]
                gene_i_idx = list(gene_i_space).index(gene_i_val)
                mut_strat = np.random.choice([0, 1])
                if mut_strat  == 1: # Replace value with the prev or next value in its gene space
                    next_idx = gene_i_idx + 1
                    prev_idx = gene_i_idx - 1
                    new_idx = np.random.choice([next_idx, prev_idx])
                    if len(gene_i_space) < 2:
                        continue
                    elif new_idx > len(gene_i_space) - 1:
                        new_idx = prev_idx
                    elif new_idx < 0:
                        new_idx = next_idx
                    offsprings[ospr_i, gene_i] = gene_i_space[new_idx]
                else: # Replace value with random value in its gene space
                    offsprings[ospr_i, gene_i] = np.random.choice(gene_i_space)
        return np.array(offsprings)
    gene_space = [[-1, -0.1, 0.1, 1] for i in range(3)]
    gene_space += [[10], [0], [0]]
    gene_space += [[0, 0.2, 0.4, 0.8] for i in range(3)]
    gene_space += [[-1, 0, 1] for i in range(6)]
    ga_instance = pygad.GA(
        fitness_func=fitness_fnc,
        num_genes=len(gene_space),
        gene_space=gene_space,
        on_generation=on_generation,
        num_generations=1000,
        sol_per_pop=12,
        num_parents_mating=8,
        keep_elitism=2,
        mutation_type=mutation_func,
        crossover_type=crossover_func,
        parent_selection_type="rank",
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    ## Save Results ##
    with open(os.path.join(save_dir, 'best_sol.json'), 'w') as f:
        json.dump(list(solution), f, indent=2)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


def evolve_ifate(save_dir):
    setup_dirs(save_dir)
    td = np.arange(0, 121, 1)
    ud = np.zeros_like(td)
    ud[40:80:10] = 1
    ud[80:121:1] = 1
    uf = interp1d(td, ud, kind='nearest')
    def fitness_fnc(params, params_idx):
        try:
            model, y0 = gen_model(params)
            tm, ym = sim_model(td, y0, uf, model)
            ym_dense = ym[1]
            ym_sparse = ym[5]
            cost_params = (np.count_nonzero(params[3:6]) + np.count_nonzero(params[9:]))/(len(params[3:6]) + len(params[9:]) + 1)
            dense_fitness = (1 - np.mean(ym_dense[0:40])) * (1 - np.mean(ym_dense[40:80])) * np.mean(ym_dense[80:121])
            sparse_fitness = (1 - np.mean(ym_sparse[0:40])) * np.mean(ym_sparse[40:80]) * (1 - np.mean(ym_sparse[80:121]))
            fitness = (dense_fitness*sparse_fitness)/(cost_params**2)
            fitness = 0 if np.isnan(fitness) else fitness
            return fitness
        except Exception as e:
            print(e)
            return 0
    ta = time.time()
    iter_ti = time.time()
    data = []
    def on_generation(ga_instance):
        nonlocal iter_ti, data
        ti = time.time()
        iter_dt = ti - iter_ti
        iter_ti = time.time()
        ti_ta = ti - ta
        xi_gen = ga_instance.generations_completed
        yi_fit = np.max(ga_instance.last_generation_fitness)
        zi_sol = list(ga_instance.last_generation_elitism[0])
        z_col_names = ['t', 'x', 'y']
        z_col_names += [
            'kra', 'krb', 'krc',
            'kua', 'kub', 'kuc',
            'kta', 'ktb', 'ktc',
            'kba', 'kab', 'kac',
            'kca', 'kcb', 'kbc']
        z_col_vals = [ti_ta, xi_gen, yi_fit] + zi_sol
        data.append(dict(zip(z_col_names, z_col_vals)))
        status = [
            ['tot_time', iter_ti - ta],
            ['seconds/iter', iter_dt],
            ['generations', xi_gen],
            ['fitness', list(ga_instance.last_generation_fitness)],
            ['best_sol', list(ga_instance.last_generation_elitism[0])]
        ]
        print(tabulate(status))
    def crossover_func(parents, offsprings_size, ga_instance):
        offsprings = []
        idx = 0
        option = np.random.choice([0, 1, 2], p=[0.9, 0.05, 0.05])
        while len(offsprings) != offsprings_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            if option == 1:
                offsprings.append(parent1)
            elif option == 2:
                offsprings.append(parent2)
            else:
                node = np.random.choice([0, 1, 2])
                child = parent1.copy()
                child[node::3] = parent2[node::3]
                offsprings.append(child)
            idx += 1
        return np.array(offsprings)
    def mutation_func(offsprings, ga_instance):
        n_genes = offsprings.shape[1]
        n_offspr = offsprings.shape[0]
        idx_genes = np.arange(n_genes)
        idx_offspr = np.arange(n_offspr)
        for ospr_i in idx_offspr:
            # For each offspring, determine which genes to mutate
            mut_op = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            if mut_op == 1: # Randomize all genes
                genes_to_mutate = idx_genes
            elif mut_op == 2: # Randomize all genes affecting a random single species
                node_k = np.random.choice([0, 1, 2])
                genes_to_mutate = idx_genes[node_k::3]
            elif mut_op == 3: # Randomize all genes of a random category
                gcats = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11, 12, 13, 14]}
                genes_to_mutate = gcats[np.random.choice([0, 1, 2, 3])]
            else:
                n_mutations = np.random.choice(idx_genes)
                genes_to_mutate = np.random.choice(idx_genes, size=n_mutations, replace=False)
            # print(genes_to_mutate)
            for gene_i in genes_to_mutate:
                gene_i_space = ga_instance.gene_space[gene_i]
                gene_i_val = offsprings[ospr_i, gene_i]
                gene_i_idx = list(gene_i_space).index(gene_i_val)
                mut_strat = np.random.choice([0, 1])
                if mut_strat  == 1: # Replace value with the prev or next value in its gene space
                    next_idx = gene_i_idx + 1
                    prev_idx = gene_i_idx - 1
                    new_idx = np.random.choice([next_idx, prev_idx])
                    if len(gene_i_space) < 2:
                        continue
                    elif new_idx > len(gene_i_space) - 1:
                        new_idx = prev_idx
                    elif new_idx < 0:
                        new_idx = next_idx
                    offsprings[ospr_i, gene_i] = gene_i_space[new_idx]
                else: # Replace value with random value in its gene space
                    offsprings[ospr_i, gene_i] = np.random.choice(gene_i_space)
        return np.array(offsprings)
    gene_space = [[-1, -0.01, 0.01, 1] for i in range(3)]
    gene_space += [[0, 10], [0, 10], [0, 10]]
    gene_space += [[0] for i in range(3)]
    gene_space += [[-10, -1, 0, 1, 10] for i in range(6)]
    ga_instance = pygad.GA(
        fitness_func=fitness_fnc,
        num_genes=len(gene_space),
        gene_space=gene_space,
        on_generation=on_generation,
        num_generations=1000,
        sol_per_pop=12,
        num_parents_mating=8,
        keep_elitism=2,
        mutation_type=mutation_func,
        crossover_type=crossover_func,
        parent_selection_type="rank",
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    ## Save Results ##
    with open(os.path.join(save_dir, 'best_sol.json'), 'w') as f:
        json.dump(list(solution), f, indent=2)
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_dir, 'y.csv'), index=False)


def plot_figure_1():
    td = np.arange(0, 121, 1)
    ud = np.zeros_like(td)
    ud[60] = 1
    uf = interp1d(td, ud, kind='nearest')
    u_df = pd.DataFrame({'t': td, 'u': ud})
    params = [
        0.0, 0.0, -1.0,
        0.0, 0.0, 10.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0]
    model, y0 = gen_model(params)
    tm, ym = sim_model(td, y0, uf, model)
    ym = ym[-1]
    y_df = pd.DataFrame({'t': tm, 'y': ym})
    save_dir = '/home/phuong/data/1-fakr/0-model/example/'
    palette = ['#DC267F']
    xlabel = 'Time'
    ylabel = r'$\mathdefault{[C_{a}]}$'
    ulabel = 'Input'
    plot_uy(y_df, u_df, save_dir, yd_df=None, fname='y.png', dpi=300, lgd_loc='best',
        xlabel=xlabel, ylabel=ylabel, ulabel=ulabel, ymin=-0.1, ymax=1.1, palette=palette)


def plot_figure_2():
    palette = ['#648FFF']
    save_dir = '/home/phuong/data/1-fakr/0-model/ifate/'
    fitness_df = pd.DataFrame()
    for i, dn in enumerate(natsorted([x[1] for x in os.walk(save_dir)][0])):
        yi_df = pd.read_csv(os.path.join(save_dir, str(dn), 'y.csv'))
        fi_df = yi_df[['x', 'y']]
        fi_df.columns = ['t', 'y']
        fi_df['h'] = np.ones_like(fi_df['y']) * i
        fitness_df = pd.concat([fitness_df, fi_df])
    plot_uy(fitness_df, u_df=None, save_dir=save_dir, fname='fitness.png', dpi=300, lgd_loc='best',
        style='h', ylabel='Fitness', xlabel='Generation', group_labels=[],
        ymin=None, ymax=None, order=None, palette=palette)
    sample_n = 2
    sample_dir = os.path.join(save_dir, str(sample_n))
    df = pd.read_csv(os.path.join(sample_dir, 'y.csv'))
    td = np.arange(0, 121, 1)
    ud = np.zeros_like(td)
    ud[40:80:10] = 1
    ud[80:121:1] = 1
    uf = interp1d(td, ud, kind='nearest')
    with open(os.path.join(sample_dir, 'best_sol.json'), 'r') as f:
        params = json.load(f)
    model, y0 = gen_model(params)
    tm, ym = sim_model(td, y0, uf, model)
    ym = ym[-1]
    plot_netw_fitn(save_dir=sample_dir, n_fit=df.x.values, y_fit=df.y.values, t_conc=tm, y_conc=ym, params=params)


def plot_oscillator_results():
    save_dir = '/home/phuong/data/1-fakr/0-model/oscillator/4/'
    td0 = np.arange(0, 301, 1)
    td1 = np.arange(0, 301, 1)
    ud0 = np.zeros_like(td0)
    ud1 = np.zeros_like(td1)
    ud1[150] = 1
    uf0 = interp1d(td0, ud0, kind='nearest')
    uf1 = interp1d(td1, ud1, kind='nearest')
    with open(os.path.join(save_dir, 'best_sol.json'), 'r') as f:
        params = json.load(f)
    model, y0 = gen_model(params)
    tm0, ym0 = sim_model(td0, y0, uf0, model)
    y1 = ym0[:, -1]
    tm, ym = sim_model(td1, y1, uf1, model)
    n_peaks0 = len(find_peaks(ym[-1][:150], prominence=0.001)[0])
    n_valls0 = len(find_peaks(-ym[-1][:150], prominence=0.001)[0])
    n_peaks1 = len(find_peaks(ym[-1][150:], prominence=0.02)[0])
    n_valls1 = len(find_peaks(-ym[-1][150:], prominence=0.02)[0])
    cost_params = (np.count_nonzero(params[6:]))/(len(params[6:]) + 1)
    fitness = (n_peaks1*n_valls1 - n_peaks0*n_valls0)/(cost_params**2)
    u_df = pd.DataFrame({'t': td1, 'u': ud1})
    ym_a_df = pd.DataFrame({'t': tm, 'y': ym[1], 'h': np.ones_like(tm)*0})
    ym_b_df = pd.DataFrame({'t': tm, 'y': ym[3], 'h': np.ones_like(tm)*1})
    ym_c_df = pd.DataFrame({'t': tm, 'y': ym[5], 'h': np.ones_like(tm)*2})
    ym_df = pd.concat([ym_a_df, ym_b_df, ym_c_df])
    palette = ['#785EF0', '#FE6100', '#DC267F']
    group_labels = [r'$\mathdefault{[A_{a}]}$', r'$\mathdefault{[B_{a}]}$', r'$\mathdefault{[C_{a}]}$']
    plot_uy(ym_df, u_df, save_dir, fname='y_sol.png', overlay_tu=False, dpi=300, lgd_loc='lower left',
        ylabel='Output', ulabel='Input', group_labels=group_labels, ymin=None, ymax=None, order=None, palette=palette)
    plot_model(save_dir, params)
    y_df = pd.read_csv(os.path.join(save_dir, 'y.csv'))
    fitness_df = y_df[['x', 'y']]
    fitness_df.columns = ['t', 'y']
    palette = ['#648FFF']
    plot_uy(fitness_df, u_df=None, save_dir=save_dir, fname='fitness.png', dpi=300, lgd_loc='best',
        ylabel='Fitness', xlabel='Generation', group_labels=[],
        ymin=None, ymax=None, order=None, palette=palette)


def plot_ifate_results():
    save_dir = '/home/phuong/data/1-fakr/0-model/ifate/4/'
    td = np.arange(0, 121, 1)
    ud = np.zeros_like(td)
    ud[40:80:10] = 1
    ud[80:121:1] = 1
    uf = interp1d(td, ud, kind='nearest')
    with open(os.path.join(save_dir, 'best_sol.json'), 'r') as f:
        params = json.load(f)
    model, y0 = gen_model(params)
    tm, ym = sim_model(td, y0, uf, model)
    ym_dense = ym[1]
    ym_sparse = ym[5]
    cost_params = (np.count_nonzero(params[3:6]) + np.count_nonzero(params[9:]))/(len(params[3:6]) + len(params[9:]) + 1)
    dense_fitness = (1 - np.mean(ym_dense[0:40])) * (1 - np.mean(ym_dense[40:80])) * np.mean(ym_dense[80:121])
    sparse_fitness = (1 - np.mean(ym_sparse[0:40])) * np.mean(ym_sparse[40:80]) * (1 - np.mean(ym_sparse[80:121]))
    fitness = (dense_fitness*sparse_fitness)/(cost_params**2)
    u_df = pd.DataFrame({'t': td, 'u': ud})
    ym_a_df = pd.DataFrame({'t': tm, 'y': ym[1], 'h': np.ones_like(tm)*0})
    ym_c_df = pd.DataFrame({'t': tm, 'y': ym[5], 'h': np.ones_like(tm)*2})
    ym_df = pd.concat([ym_a_df, ym_c_df])
    palette = ['#785EF0', '#DC267F']
    group_labels = ['Dense Ch', 'Sparse Ch']
    plot_uy(ym_df, u_df, save_dir, fname='y_sol.png', style=False, overlay_tu=False, dpi=300, lgd_loc='upper left',
        ylabel='Output', ulabel='Input', group_labels=group_labels, ymin=-0.1, ymax=1.1, order=None, palette=palette)
    plot_model(save_dir, params)
    y_df = pd.read_csv(os.path.join(save_dir, 'y.csv'))
    fitness_df = y_df[['x', 'y']]
    fitness_df.columns = ['t', 'y']
    palette = ['#648FFF']
    plot_uy(fitness_df, u_df=None, save_dir=save_dir, fname='fitness.png', dpi=300, lgd_loc='best',
        ylabel='Fitness', xlabel='Generation', group_labels=[],
        ymin=None, ymax=None, order=None, palette=palette)


def plot_all_fitness(save_dir):
    palette = ['#648FFF']
    fitness_df = pd.DataFrame()
    for i, dn in enumerate(natsorted([x[1] for x in os.walk(save_dir)][0])):
        yi_df = pd.read_csv(os.path.join(save_dir, str(dn), 'y.csv'))
        fi_df = yi_df[['x', 'y']]
        fi_df.columns = ['t', 'y']
        fi_df['h'] = np.ones_like(fi_df['y']) * i
        fitness_df = pd.concat([fitness_df, fi_df])
    plot_uy(fitness_df, u_df=None, save_dir=save_dir, fname='fitness.png', style='h', dpi=300, lgd_loc='best',
        ylabel='Fitness', xlabel='Generation', group_labels=[],
        ymin=None, ymax=None, order=None, palette=palette)


def plot_osc_vid(root_dir):
    root_dir = '/home/phuong/data/1-fakr/0-model/oscillator/3/'
    save_dir = os.path.join(root_dir, 'sample')
    setup_dirs(save_dir)
    y_df = pd.read_csv(os.path.join(root_dir, 'y.csv'))
    for i in range(len(y_df)):
        if i > 25:
            break
        n_fit = y_df['t'][:i+1].values
        y_fit = y_df['y'][:i+1].values
        params = y_df.loc[i, 'kra':]
        td0 = np.arange(0, 301, 1)
        td1 = np.arange(0, 301, 1)
        ud0 = np.zeros_like(td0)
        ud1 = np.zeros_like(td1)
        ud1[150] = 1
        uf0 = interp1d(td0, ud0, kind='nearest')
        uf1 = interp1d(td1, ud1, kind='nearest')
        model, y0 = gen_model(params)
        tm0, ym0 = sim_model(td0, y0, uf0, model)
        y1 = ym0[:, -1]
        tm, ym = sim_model(td1, y1, uf1, model)
        ym = ym[-1]
        plot_netw_fitn(save_dir, n_fit, y_fit, tm, ym, params)


if __name__ == '__main__':
    save_dir = '/home/phuong/data/1-fakr/0-model/osc_vid'
    for n in range(1):
        save_dir_n = os.path.join(save_dir, str(n))
        evolve_oscillator(save_dir_n)
    plot_oscillator_results()
    plot_all_fitness(save_dir)

    # save_dir = '/home/phuong/data/1-fakr/0-model/ifate/'
    # for n in range(5):
    #     save_dir_n = os.path.join(save_dir, str(n))
    #     evolve_ifate(save_dir_n)
    # plot_ifate_results()
    # plot_all_fitness(save_dir)



    # plot_figure_1()

    # plot_figure_2()

    # plot_osc_vid('/home/phuong/data/1-fakr/0-model/osc_vid/')
