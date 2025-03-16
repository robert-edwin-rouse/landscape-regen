#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:10:57 2025

@author: robertrouse
"""

import numpy as np
import torch
from dataclasses import dataclass
from pymoo.core.problem import ElementwiseProblem


@dataclass
class OptimizerConfig:
    population_size: int = 500
    offspring: int = 20
    crossover_probability: float = 0.8
    crossover_eta: float = 16
    mutation_eta: float = 10
    max_generations: int = 2000
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class LandscapeOptimisation(ElementwiseProblem):
    def __init__(self, model):
        self.model = model
        super().__init__(n_var=8,
                         n_obj=3,
                         n_ieq_constr=17,
                         xl=np.array([0,0,0,0,0,0,0,0]),
                         xu=np.array([1,1,1,1,1,0.35,1,1]))

    def _evaluate(self, x, out, *args, **kwargs):
        z = torch.from_numpy(x)
        z = self.model(z.float()).data.numpy()
        out["F"] = [z[0], -z[1], -z[2]]
        out["G"] = self._calculate_constraints(x, z)
    
    def _calculate_constraints(self, x, z):
        constraints = [x[0] + x[1] - 1,
                       x[2]*0.02781 + x[0] - 1,
                       x[2]*0.02701 + x[1] - 1,
                       x[4]*0.3799 + x[0] - 1,
                       x[4]*0.3690 + x[1] - 1,
                       x[4]*0.4815 + x[6] - 1,
                       x[5]*0.3667 + x[0] - 1,
                       x[5]*0.3561 + x[1] - 1,
                       x[5]*0.4648 + x[6] - 1,
                       x[5]*0.7934 + x[7] - 1,
                       x[0]*1.3393 + x[6] - 1.3393,
                       x[1]*1.3791 + x[6] - 1.3791,
                       x[7]*0.5858 + x[6] - 1.0818,
                       x[0]*1.2298 + x[7] - 1.3617,
                       x[1]*1.2065 + x[7] - 1.3384,
                       0 - z[1],
                       -1 - z[0],]
        return constraints