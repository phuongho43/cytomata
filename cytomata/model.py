import os
import time

import numpy as np
from scipy.integrate import ode
from scipy.stats.qmc import Halton
from tabulate import tabulate
from joblib import Parallel, delayed


def sim_lov(tt, X0, uu, kk):
    def model(t, X):
        [Ai, Aa, B, AB] = X
        [kl, kd, kb] = kk
        v = 0 if u != 0 else 1
        dAi = -u*kl*Ai + v*kd*Aa - kb*Ai*B
        dAa = u*kl*Ai - v*kd*Aa + u*kl*AB
        dB = -kb*Ai*B + u*kl*AB
        dAB = kb*Ai*B - u*kl*AB
        return [dAi, dAa, dB, dAB]
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', rtol=1e-6, atol=1e-6, max_step=0.1)
    solver.set_initial_value(X0)
    sol_t = [tt[0]]
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_X.append(solver.y)
    return np.array(sol_t), np.array(sol_X).T


def sim_ilid(tt, X0, uu, kk):
    def model(t, X):
        [Ai, Aa, B, AB] = X
        [kl, kd, kb] = kk
        v = 0 if u != 0 else 1
        dAi = -u*kl*Ai + v*kd*Aa + v*kd*AB
        dAa = u*kl*Ai - v*kd*Aa - kb*Aa*B
        dB = -kb*Aa*B + v*kd*AB
        dAB = kb*Aa*B - v*kd*AB
        return [dAi, dAa, dB, dAB]
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', rtol=1e-6, atol=1e-6, max_step=0.1)
    solver.set_initial_value(X0)
    sol_t = [tt[0]]
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_X.append(solver.y)
    return np.array(sol_t), np.array(sol_X).T


def sim_sparser(tt, X0, uu, kk):
    def model(t, X):
        [Ai0, Aa0, Bi0, Ba0, C0, AiBi0, AiBa0, BaC0, AiBaC0] = X0
        [Ai, Aa, Bi, Ba, C, AiBi, AiBa, BaC, AiBaC] = X
        [kl1, kd1, kb1, kl2, kd2, kb2] = kk
        v = 0 if u != 0 else 1
        dAi = -u*kl1*Ai + v*kd1*Aa - kb1*Ai*Bi - kb1*Ai*Ba - kb1*Ai*BaC
        dAa = u*kl1*Ai - v*kd1*Aa + u*kl1*AiBi + u*kl1*AiBa + u*kl1*AiBaC
        dBi = -u*kl2*Bi + v*kd2*Ba - kb1*Ai*Bi + v*kd2*BaC
        dBa = u*kl2*Bi - v*kd2*Ba - kb1*Ai*Ba - kb2*Ba*C + u*kl1*AiBi + u*kl1*AiBa
        dC = -kb2*Ba*C - kb2*AiBa*C + v*kd2*BaC + v*kd2*(AiBaC-AiBaC0)
        dAiBi = kb1*Ai*Bi -u*kl1*AiBi + v*kd2*(AiBaC-AiBaC0)
        dAiBa = kb1*Ai*Ba - kb2*AiBa*C - u*kl1*AiBa
        dBaC = kb2*Ba*C - kb1*Ai*BaC - v*kd2*BaC + u*kl1*AiBaC
        dAiBaC = kb2*AiBa*C + kb1*Ai*BaC - v*kd2*(AiBaC-AiBaC0) - u*kl1*AiBaC
        return [dAi, dAa, dBi, dBa, dC, dAiBi, dAiBa, dBaC, dAiBaC]
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', rtol=1e-6, atol=1e-6, max_step=0.1)
    solver.set_initial_value(X0)
    sol_t = [tt[0]]
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_X.append(solver.y)
    return np.array(sol_t), np.array(sol_X).T


def sim_signet(tt, uu, kr, ku, kX, kt):
    """Simulates the dynamic response of the SigNet ODE model with N nodes.

    Args:
        tt (1 x t array): t equally spaced timepoints
        uu (1 x t array): input stimuli intensity/conc/amount at each timepoint
        kr (1 x N array): reversion rates of each node
        ku (1 x N array): induction rates of stimuli towards each node
        kX (N x N array): interaction rates of each node towards each other
        kt (N x N array): threshold conc/amount for each effector node

    Returns:
        sol_t (1D array): timepoints of the simulation (same as tt)
        sol_X (2D array): N x t array of the dynamic response for each node
    """
    # reversion rates dictate whether the initial value of a node is 0 or 1
    X0 = np.where(kr < 0, 0, 1)
    # induction direction is always opposite of reversion direction
    ku = np.where(kr*ku > 0, -ku, ku)
    def model(t, X):
        # reaction towards a node depends on that node's current conc/amount
        Xr = np.where(kr < 0, X, 1 - X)
        Xu = np.where(ku < 0, X, 1 - X)
        Xy = np.where(kX.T < 0, X, 1 - X).T
        # conc threshold: X_ij is equal to 0 until it exceeds kt_ij
        Yt = np.where(X - kt < 0, 0, X - kt)
        # dx/dt = reversion + induction + interaction
        dX = kr*Xr + ku*Xu*u + np.sum(kX*Xy*Yt, axis=1)
        return dX
    solver = ode(model)
    solver.set_integrator('vode', method='bdf', rtol=1e-6, atol=1e-6, max_step=0.1)
    solver.set_initial_value(X0)
    sol_t = [tt[0]]
    sol_X = [X0]
    for i in range(1, len(tt)):
        u = uu[i]
        # v = np.where(u != 0, 0, 1)
        solver.integrate(tt[i])
        sol_t.append(solver.t)
        sol_X.append(solver.y)
    return np.array(sol_t), np.array(sol_X).T


def calc_hypervolume2D(pf_obj, ref):
    """Calculate the hypervolume indicator for an optimization problem with 2 objectives.

    Args:
        pf_obj (2D array): array of m nondominated front (front 0) individuals by j=2 objectives
        ref (1D array): array of size 2 specifying reference point for HV calculation.
            ref value 1: the minimum value possible for objective 1
            ref value 2: the minimum value possible for objective 2

    Returns:
        hv (float): hypervolume value
    """
    pf_obj = np.unique(pf_obj, axis=0)  # get unique obj scores and sort by obj 1
    df1 = np.diff([ref[0]] + list(pf_obj[:, 0]))  # rectangle widths
    df2 = np.abs(pf_obj[:, 1] - ref[1])  # rectangle heights
    hv = (df1*df2).sum()
    return hv


class NSGAII(object):
    """Non-dominated Sorting Genetic Algorithm II multi-objective optimizer.

    Implemented based on:
    Deb, Kalyanmoy, et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II."
    IEEE transactions on evolutionary computation 6.2 (2002): 182-197.

    Attributes:
        obj_func (func): Given an individual (1D array of n parameters), calculates a 1D array of j
            objective scores representing the fitness of the individual.
            IMPORTANT: We assume a *maximization* scheme for every objective score.
        param_space (list of array-likes): List of n array-likes where n is equal to the number of
            parameters in each individual
        pop_size (int): number of individuals (m) in the population
        RNG (obj): numpy random number generator
        population (2D array): random population of m individuals by n parameters
        obj_scores (2D array): array of m individuals by j objective scores
        fronts (list of lists): list of objective fronts where each front is a list of indices
            corresponding to individuals in the population. The first front (front[0])
            corresponds to the current best individuals.
        ranks (1D array): the front number corresponding to each individual in the population.
            Lower numbered rank is better e.g. rank 0 corresponds to front[0].
        cdists (1D array): the crowding distance corresponding to each individual in the population.
                Higher crowding distance is better (more diverse/unique).
    """
    def __init__(self, obj_func, param_space, pop_size, rng_seed=None):
        self.RNG = np.random.default_rng(rng_seed)
        self.halton = Halton(d=len(param_space))
        self.obj_func = obj_func
        self.param_space = param_space
        self.pop_size = pop_size
        self.population = self.create_random_population(pop_size)

    def create_random_population(self, m):
        """Initialize a random population by picking values randomly from param_space.
        """
        lb = [0 for k_space in self.param_space]
        ub = [len(k_space) for k_space in self.param_space]
        sample = self.halton.integers(l_bounds=lb, u_bounds=ub, n=m)
        randoms = np.empty((m, len(self.param_space)))
        for n in range(len(self.param_space)):
            k_space = list(np.array(self.param_space[n]))
            randoms[:, n] = np.take(k_space, sample[:, n])
        return randoms

    def eval_objective(self, population):
        """Evaluate a population based on the objective function (obj_func).
        
        Args:
            population (2D array): population of m individuals by n parameters

        Returns:
            obj_scores (2D array): array of m individuals by j objective scores
        """
        obj_scores = Parallel(n_jobs=os.cpu_count())(
            delayed(self.obj_func)(indiv)
            for indiv in population
        )
        return np.array(obj_scores)

    def dominates(self, p_obj, q_obj):
        """Evaluates whether individual p dominates individual q.

        Individual p dominates individual q if p is no worse than q in all objectives and p is
        strictly better than q in at least one objective.

        Args:
            p_obj (1D array-like): array of j objective scores corresponding to individual p
            q_obj (1D array-like): array of j objective scores corresponding to individual q

        Returns:
            True if p dominates q else False
        """
        if np.all(p_obj >= q_obj) and np.any(p_obj > q_obj):
            return True
        else:
            return False

    def sort_fronts_ndom(self):
        """Sort individuals into objective fronts.

        For multi-objective optimization, individuals are ranked and sorted into fronts based on
        dominance. This ranking is later used for selecting parents and survivors.
        """
        pop_idx = range(len(self.obj_scores))
        S = [[] for i in pop_idx]  # set of other indivs that a given indiv dominates
        n = [0 for i in pop_idx]  # count of indivs that dominates a given indiv
        F = [[]]  # fronts
        R = [0 for i in pop_idx]  # ranks
        # assign domination set and counts for every individual
        for p in pop_idx:
            for q in pop_idx:
                if self.dominates(self.obj_scores[p], self.obj_scores[q]):
                    S[p].append(q)
                elif self.dominates(self.obj_scores[q], self.obj_scores[p]):
                    n[p] += 1
        # sort individuals into obj fronts based on domination set and counts
            if n[p] == 0:
                F[0].append(p)
                R[p] = 0
        i = 0
        while F[i]:
            Q = []
            for p in F[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        Q.append(q)
                        R[q] = i + 1
            i += 1
            F.append(Q)
        self.fronts = F[:-1]  # last front is empty --> discard
        self.ranks = np.array(R)

    def calc_crowd_dist(self):
        """Calculate the crowding distance for each individual.

        The distance between an individual's two closest neighbors in the same front. During parent
        or survival selection, this metric is used to distinguish between individuals when they have
        the same front/rank.
        """
        cdists = [0 for p in range(len(self.obj_scores))]
        for front in self.fronts:
            for j in range(self.obj_scores.shape[1]):
                # sort front based on objective j
                front_j = sorted(front, key=lambda m: self.obj_scores[m, j])
                # assign high cdist to the most fringe individuals
                cdists[front_j[0]] = np.inf
                cdists[front_j[-1]] = np.inf
                for i in range(1, len(front) - 1):
                    # normalized objective space distance between neighbors
                    m_dist = self.obj_scores[front_j[i+1], j] - self.obj_scores[front_j[i-1], j]
                    m_range = np.max(self.obj_scores[:, j]) - np.min(self.obj_scores[:, j])
                    if m_range == 0:
                        cdists[front_j[i]] += 0
                    else:
                        cdists[front_j[i]] += m_dist/m_range
        self.cdists = np.array(cdists)

    def select_parent(self):
        """Select a candidate for mating using binary tournament selection.

        NSGA-II version of binary tournament selection that takes into account objective front
        ranking and crowding distance.

        Returns:
            candidate (1D array): candidate individual of n parameters
        """
        pop_idx = range(len(self.population))
        c1_idx, c2_idx = self.RNG.choice(pop_idx, 2, replace=False)
        winner_idx = self.RNG.choice([c1_idx, c2_idx])
        # Choose based on better rank
        c1_rank = self.ranks[c1_idx]
        c2_rank = self.ranks[c2_idx]
        if c1_rank < c2_rank:
            winner_idx = c1_idx
        elif c2_rank < c1_rank:
            winner_idx = c2_idx
        # If equal rank than choose based on better crowding distance
        else:
            c1_cdist = self.cdists[c1_idx]
            c2_cdist = self.cdists[c2_idx]
            if c1_cdist > c2_cdist:
                winner_idx = c1_idx
            elif c2_cdist > c1_cdist:
                winner_idx = c2_idx
        # If equal crowding distance, choose one of them randomly
        return self.population[winner_idx]

    def recombine(self, parent1, parent2, rec_rate):
        """Produce an offspring by inheriting params from two parents.

        For each parameter position, inherit parameter from parent 2 with probability rec_rate.

        Args:
            parent1 (1D array): candidate individual #1 containing n parameters
            parent2 (1D array): candidate individual #2 containing n parameters
            rec_rate (float): chance between [0, 1] of inheriting each parameter from parent 2
                and the rest from parent 1. Since parent 1 and 2 are interchangeable, the effective
                range is [0, 0.5]

        Returns:
            recomb (1D array): resulting child individual containing n parameters
        """
        recomb = parent1.copy()
        for n in range(len(recomb)):
            if self.RNG.random() <= rec_rate:
                recomb[n] = parent2[n]
        return recomb

    def mutate(self, original, mut_rate, mut_spread):
        """Mutate each param of an individual.

        Args:
            original (1D array): individual containing n parameters
            mut_rate (float): chance between [0, 1] of mutating each parameter position
            mut_spread (float): probability between [0, 1] for the geometric distribution used for
                sampling mutations. Lower = more chance to mutate to a value far away from current.

        Returns:
            mutant (1D array): resulting mutant individual containing n parameters
        """
        mutant = original.copy()
        for n in range(len(mutant)):
            if self.RNG.random() <= mut_rate:
                k_space = list(np.array(self.param_space[n]))
                k_idx = k_space.index(mutant[n])
                mut_type = self.RNG.choice(['increment', 'decrement'])
                if mut_type == 'increment' and k_idx != len(k_space) - 1:
                    step = min(self.RNG.geometric(p=mut_spread), len(k_space) - k_idx - 1)
                    mutant[n] = k_space[k_idx + step]
                elif mut_type == 'decrement' and k_idx != 0:
                    step = min(self.RNG.geometric(p=mut_spread), k_idx)
                    mutant[n] = k_space[k_idx - step]
        return mutant

    def produce_children(self, rec_rate, mut_rate, mut_spread):
        """Perform process of selecting 2 parents, producing child, and introducing mutations.

        Args:
            rec_rate (float): chance between [0, 1] of inheriting each parameter from parent 2
                and the rest from parent 1. Since parent 1 and 2 are interchangeable, the effective
                range is [0, 0.5]
            mut_rate (float): chance between [0, 1] of mutating each parameter
            mut_spread (float): probability between [0, 1] for the geometric distribution used for
                sampling mutations. Lower = more chance to mutate to a value far away from current.

        Returns:
            children (2D array): population of pop_size (m) child individuals by n parameters
        """
        children = []
        while len(children) < self.pop_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            recomb = self.recombine(parent1, parent2, rec_rate)
            mutant = self.mutate(recomb, mut_rate, mut_spread)
            children.append(mutant)
        return np.array(children)

    def select_survivors(self):
        """NSGA-II survivor selection method.

        Apply towards combined parents + children population to reduce the total number of
        individuals down to original population size.
        """
        new_pop = []
        new_obj = []
        new_cdists = []
        new_fronts = []
        new_ranks = []
        i = 0  # iterate through obj fronts from best to worst
        while len(new_pop) < self.pop_size:
            curr_front = self.fronts[i]
            curr_pop_size = len(new_pop)
            curr_front_size = len(curr_front)
            projected_size = curr_pop_size + curr_front_size
            # prioritize individuals with better rank/front
            if projected_size <= self.pop_size:
                new_pop += list(self.population[curr_front])
                new_obj += list(self.obj_scores[curr_front])
                new_cdists += list(self.cdists[curr_front])
                new_ranks += list(self.ranks[curr_front])
                new_fronts.append(list(range(curr_pop_size, projected_size)))
            # when a front exceeds pop_size, prioritize individuals with better crowding distance
            else:
                curr_front_sorted = sorted(curr_front, key=lambda x: self.cdists[x])
                n_last_spots = self.pop_size - curr_pop_size
                ls_idx = curr_front_sorted[-n_last_spots:]
                new_pop += list(self.population[ls_idx])
                new_obj += list(self.obj_scores[ls_idx])
                new_cdists += list(self.cdists[ls_idx])
                new_ranks += list(self.ranks[ls_idx])
                new_fronts.append(list(range(curr_pop_size, self.pop_size)))
            i += 1
        self.population = np.array(new_pop)
        self.obj_scores = np.array(new_obj)
        self.ranks = np.array(new_ranks)
        self.cdists = np.array(new_cdists)
        self.fronts = new_fronts

    def evolve(self, rec_rate=0.1, mut_rate=0.1, mut_spread=0.5, n_gen=500, hv_func=None, hv_ref=None):
        """Evolutionary optimization loop.

        Args:
            rec_rate (float): chance between [0, 1] of inheriting each parameter from parent 2
                and the rest from parent 1. Since parent 1 and 2 are interchangeable, the effective
                range is [0, 0.5]
            mut_rate (float): chance between [0, 1] of mutating each parameter
            mut_spread (float): probability between [0, 1] for the geometric distribution used for
                sampling mutations. Lower = more chance to mutate to a value far away from current.
            n_gen (int): number of generations/loops to run optimization
            hv_func (func): function to calculate hypervolume indicator
                pf_obj (2D array): nondominated front (front 0) objective scores
                ref (1D array): reference point for HV calculation
            hv_ref (1D array): reference point for hv_func

        Returns:
            data (list of dicts): list of dict data values to save at each generation
        """
        time_0 = time.time()
        data = []
        self.obj_scores = self.eval_objective(self.population)
        self.sort_fronts_ndom()
        self.calc_crowd_dist()
        for gen_i in range(n_gen):
            # display/save data
            time_i = time.time() - time_0
            data_i = {'elapsed_time': time_i, 'generation': gen_i}
            status_i = [['elapsed_time', time_i], ['generation', gen_i]]
            if self.obj_scores.shape[1] > 1:
                data_i['objective'] = self.obj_scores.tolist()
                data_i['population'] = self.population.tolist()
                status_i.append(['max_obj', np.max(self.obj_scores, axis=0)])
            else:
                obj = self.obj_scores.ravel()
                idx = np.argmax(obj)
                max_obj = obj[idx]
                max_pop = list(self.population[idx])
                data_i['objective'] = max_obj
                data_i['population'] = max_pop
                status_i.append(['max_obj', max_obj])
                status_i.append(['max_pop', max_pop])
            if hv_func is not None and hv_ref is not None:
                pf_obj = self.obj_scores[self.fronts[0]]
                hv_i = hv_func(pf_obj, hv_ref)
                data_i['hypervolume'] = hv_i
                status_i.append(['hypervolume', hv_i])
            data.append(data_i)
            print(tabulate(status_i))
            # GA steps
            children_pop = self.produce_children(rec_rate, mut_rate, mut_spread)
            children_obj = self.eval_objective(children_pop)
            self.population = np.vstack((self.population, children_pop))
            self.obj_scores = np.vstack((self.obj_scores, children_obj))
            self.sort_fronts_ndom()
            self.calc_crowd_dist()
            self.select_survivors()
        return data
