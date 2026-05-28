# -*- coding: utf-8 -*-
"""
Travelling Salesman Problem (TSP)
Student: Haoqi Tan
Course: AI52

"""

import numpy as np
import random
import math
import matplotlib.pyplot as mlt

def tsp_cost(path, dist_matrix):
    cost = 0
    n = len(path)
    for i in range(n):
        cost += dist_matrix[path[i]][path[(i + 1) % n]]
    return cost

def swap_mutation(path):
    i, j = random.sample(range(len(path)), 2)
    new_path = path.copy()
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def simulated_annealing(dist_matrix, T_init=1000, cooling_rate=0.995, iterations_per_T=200):
    n = len(dist_matrix)

    #initial solution
    current_path = list(range(n))
    random.shuffle(current_path)
    current_cost = tsp_cost(current_path, dist_matrix)

    best_path = current_path
    best_cost = current_cost
    
    T = T_init
    cost_history = []

    while T > 1e-3:
        for _ in range(iterations_per_T):
            new_path = swap_mutation(current_path)
            new_cost = tsp_cost(new_path, dist_matrix)
            
            # Acceptance condition
            if new_cost < current_cost:
                current_path = new_path
                current_cost = new_cost
            else:
                # Metropolis
                if random.random() < math.exp((current_cost - new_cost) / T):
                    current_path = new_path
                    current_cost = new_cost

            # Update the global best solution
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = current_path
        
        cost_history.append(best_cost)
        T *= cooling_rate  # Cooling
    
    return best_path, best_cost, cost_history


