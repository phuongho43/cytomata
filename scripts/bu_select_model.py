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
from natsort import natsorted
from tabulate import tabulate
import pygad

from cytomata.process import process_fit_data
from cytomata.model import sim_ilid, sim_lov, sim_sparser, gen_model, sim_model
from cytomata.plot import plot_class_group, plot_uy
from cytomata.utils import setup_dirs, clear_screen, custom_styles, custom_palette


def evolve_model():
    t = np.arange(0, 91, 1)
    u = np.zeros_like(t)
    u[30:61:10] = 1
    u[60:91:1] = 1
    uf = interp1d(t, u, kind='nearest')
    yd = np.zeros_like(t)
    yd[30:61] = 1
    def fitness_fnc(params, params_idx):
        try:
            model, y0 = gen_model(params)
            tm, ym = sim_model(t, y0, uf, params, model)
            ym = ym[-1]
            fitness = 90 - np.sum(np.abs(yd - ym))
            # fitness = (30 - np.sum(np.abs(yd[30:61] - ym[30:61])))/2
            # fitness = 60 - np.sum(ym[0:61])
            # fitness += np.sum(ym[60:91])
            fitness += 2*np.count_nonzero(params==0)
            fitness = 0 if fitness > 250 else fitness
            fitness = 0 if np.isnan(fitness) else fitness
            return fitness
        except Exception:
            return 0
    ta = time.time()
    iter_ti = time.time()
    def on_generation(ga_instance):
        nonlocal iter_ti
        ti = time.time()
        iter_dt = ti - iter_ti
        iter_ti = time.time()
        status = [
            ['tot_time', iter_ti - ta],
            ['seconds/iter', iter_dt],
            ['generations', ga_instance.generations_completed],
            ['fitness', list(ga_instance.last_generation_fitness)],
            ['best_sol', list(ga_instance.last_generation_elitism[0])]
        ]
        print(tabulate(status))
    def crossover_func(parents, offspring_size, ga_instance):
        offspring = []
        idx = 0
        while len(offspring) != offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
            node = np.random.choice([0, 1, 2])
            parent1[node::3] = parent2[node::3]
            offspring.append(parent1)
            idx += 1
        return np.array(offspring)
    def mutation_func(offspring, ga_instance):
        for chromosome_idx in range(offspring.shape[0]):
            for i in range(np.random.choice(range(offspring.shape[1]))):
                random_gene_idx = np.random.choice(range(offspring.shape[1]))
                offspring[chromosome_idx, random_gene_idx] = np.random.choice(ga_instance.gene_space[random_gene_idx])
        return offspring
    gene_space = [[-1, -0.01, 0.01, 1] for i in range(3)]
    gene_space += [[0, 10, 100], [0, 10, 100], [0, 10, 100]]
    gene_space += [[-10, -1, 0, 1, 10] for i in range(9)]
    ga_instance = pygad.GA(
        fitness_func=fitness_fnc,
        num_genes=len(gene_space),
        gene_space=gene_space,
        on_generation=on_generation,
        num_generations=5000,
        sol_per_pop=16,
        num_parents_mating=8,
        keep_elitism=2,
        mutation_type=mutation_func,
        crossover_type=crossover_func,
        parent_selection_type="rank",
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=list(solution)))


if __name__ == '__main__':
    # t = np.arange(0, 91, 1)
    # u = np.zeros_like(t)
    # u[30:61:10] = 1
    # u[60:91:1] = 1
    # uf = interp1d(t, u, kind='nearest')
    # yd = np.zeros_like(t)
    # yd[30:61] = 1
    # params = [1.0, -0.01, -0.01, 100.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 1.0]
    # model, y0 = gen_model(params)
    # t, y = sim_model(t, y0, uf, params, model)
    # plt.plot(t, u, label='BL', color='#648FFF')
    # # plt.plot(t, y[1], label='Aa', color='#785EF0')
    # # plt.plot(t, y[3], label='Ba', color='#29CA6E')
    # plt.plot(t, y[-1], label='Ca', color='#DC267F')
    # plt.plot(t, yd, label='yd', color='#34495E')
    # plt.show()
    
    evolve_model()
