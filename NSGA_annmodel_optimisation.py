#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:14:41 2024

@author: robertrouse
"""

import pandas as pd
import numpy as np
import torch
import surrogate as sr
import optimiser as op
import matplotlib.pyplot as plt
import matplotlib.ticker as mtk
from apollo import mechanics as ma
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume
from pymoo.decomposition.asf import ASF


### Set plotting style parameters
ma.textstyle()


### Data import, feature-target identification, and datasplit
df  = pd.read_csv('data/miniLUSP_output.csv')
columns = df.columns.tolist()
features = columns[1:9]
targets = columns[9:]
xspace = ma.featurelocator(df, features)
yspace = ma.featurelocator(df, targets)


config = op.OptimizerConfig()
net = torch.load('model.pt')
problem = op.LandscapeOptimisation(net)

algorithm = NSGA2(
    pop_size=config.population_size,
    n_offsprings=config.offspring,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=config.crossover_probability, eta=config.crossover_eta),
    mutation=PM(eta=config.mutation_eta),
    eliminate_duplicates=True
)
termination = get_termination("n_gen", 2000)

res = minimize(problem,
               algorithm,
               termination,
               seed=config.random_seed,
               save_history=True,
               verbose=True)

X = res.X
F = res.F
hist = res.history

n_evals = []             # corresponding number of function evaluations\
hist_F = []              # the objective space values in each generation
hist_cv = []             # constraint violation in each generation
hist_cv_avg = []         # average constraint violation in the whole population

for algo in hist:

    # store the number of function evaluations
    n_evals.append(algo.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algo.opt

    # store the least contraint violation and the average in each population
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())

    # filter out only the feasible and append and objective space values
    feas = np.where(opt.get("feasible"))[0]
    hist_F.append(opt.get("F")[feas])

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

metric = Hypervolume(ref_point= np.array([1.1, 1.1, 1.1]),
                      norm_ref_point=False,
                      zero_to_one=True,
                      ideal=approx_ideal,
                      nadir=approx_nadir)

hv = [metric.do(_F) for _F in hist_F]

plt.figure(figsize=(7, 5))
plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Hypervolume")
plt.show()



y0 = F[:,0]
y1 = -1*F[:,1]
y2 = -1*F[:,2]

def pareto_2D_slice(x, y, xlimits, ylimits, xlabel, ylabel, c, c_l):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.scatter(x, y, marker='o', s=16, c=c)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.xaxis.set_major_locator(mtk.MaxNLocator(5))
    ax.yaxis.set_major_locator(mtk.MaxNLocator(5))
    cbar = plt.colorbar(im)
    cbar.set_label(c_l)
    plt.show()

def pareto_overlay(x, y, xlimits, ylimits, xlabel, ylabel, c, c_l, l_x, l_y):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(l_x, l_y, marker='x', s=16, c='black', alpha=0.8)
    im = ax.scatter(x, y, marker='o', s=16, c=c)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.xaxis.set_major_locator(mtk.MaxNLocator(5))
    ax.yaxis.set_major_locator(mtk.MaxNLocator(5))
    cbar = plt.colorbar(im)
    cbar.set_label(c_l)
    plt.show()

y_str0 = 'CO'+r'$_2$'+'eq Emissions Relative to 2015'
y_str1 = 'Agriculatural Yield Relative to 2015'
y_str2 = 'Geometric Change in Bird Populations'



pareto_2D_slice(y0, y1, [-1.05, 1], [0.2, 1], y_str0, y_str1, y2, y_str2)
pareto_2D_slice(y0, y2, [-1.05, 1], [0.95, 1.2], y_str0, y_str2, y1,y_str1)
pareto_2D_slice(y1, y2, [0.2, 1], [0.95, 1.2],  y_str1, y_str2, y0, y_str0)

pareto_2D_slice(df['gwp_rel'], df['food_rel'], [-1.05, 1], [0.2, 1],
                y_str0, y_str1, df['birds_rel'], y_str2)
pareto_2D_slice(df['gwp_rel'], df['birds_rel'], [-1.05, 1], [0.95, 1.2],
                y_str0, y_str2, df['food_rel'], y_str1)
pareto_2D_slice(df['food_rel'], df['birds_rel'], [0.2, 1], [0.95, 1.2],
                y_str1, y_str2, df['gwp_rel'], y_str0)

pareto_overlay(y0, y1, [-1.05, 1], [0.2, 1], y_str0, y_str1, y2, y_str2, 
               df['gwp_rel'], df['food_rel'])
pareto_overlay(y0, y2, [-1.05, 1], [0.95, 1.2], y_str0, y_str2, y1,y_str1, 
               df['gwp_rel'], df['birds_rel'])
pareto_overlay(y1, y2, [0.2, 1], [0.95, 1.2],  y_str1, y_str2, y0, y_str0, 
               df['food_rel'], df['birds_rel'])




fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter(F[:,1], F[:,2], F[:,0], c=F[:,2])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter(F[:,1], F[:,0], F[:,2], c=F[:,0])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter(F[:,2], F[:,0], F[:,1], c=F[:,1])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection="3d") 
ax.scatter(F[:,0], F[:,2], F[:,1], c=F[:,1])
plt.show()

def enviro_opt(X, F, target_min, target_col):
    cache = np.hstack((X, F))
    agr_constrained = cache[np.where(cache[:,target_col]<(target_min*-1))]
    new_X = agr_constrained[:,0:X.shape[1]]
    new_F = agr_constrained[:,X.shape[1]:]
    return new_X, new_F

new_x, new_f = enviro_opt(X, F, 0.9, 9)
approx_ideal = new_f.min(axis=0)
approx_nadir = new_f.max(axis=0)
nF = (new_f-approx_ideal)/(approx_nadir-approx_ideal)
weights = np.array([0.5, 0, 0.5])
decomp = ASF()
i = decomp.do(nF, 1/weights).argmin()

full = np.hstack((X, F))
col_list = df.columns.to_list()[1:]
full_pareto = pd.DataFrame(full, columns=col_list)

full_pareto['food_rel'] = full_pareto['food_rel']*-1
full_pareto['birds_rel'] = full_pareto['birds_rel']*-1
partial_pareto = full_pareto[full_pareto.food_rel>0.9]

solution_set = pd.DataFrame(np.hstack((X,F)), columns=df.columns.to_list()[1:])
solution_set['food_rel'] = solution_set['food_rel']*-1
solution_set['birds_rel'] = solution_set['birds_rel']*-1
solution_set.to_csv('Pareto.csv')