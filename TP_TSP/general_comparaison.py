# -*- coding: utf-8 -*-
"""
Travelling Salesman Problem (TSP)
Student: Haoqi Tan
Course: AI52

"""

import time
import numpy as np
import pandas as pd
from sa import simulated_annealing
from ga import genetic_algorithm
from qga import quantum_genetic_algorithm
from aco import ant_colony_optimization
from exact_solver import solve_exact_tsp   

from instances import generate_cities, compute_distance_matrix


# =============================
#  6 TSP instances
# =============================
instances = [
    5,
    8,
    10,
    12,
    15,
    20
]


# =============================
#  run all the algorithms
# =============================
results = []

for n in instances:

    print("\n============================")
    print(f"🔵 Running instance with {n} cities")
    print("============================")

    coords = generate_cities(n, seed=42 + n)
    dist_matrix = compute_distance_matrix(coords)

    row = {
        "n": n
    }

    # ------- SA -------
    t0 = time.time()
    sa_path, sa_cost, _ = simulated_annealing(dist_matrix)
    t1 = time.time()
    row["SA_cost"] = sa_cost
    row["SA_time"] = round(t1 - t0, 4)

    print(f"SA   : cost={sa_cost:.2f}, time={row['SA_time']} sec")

    # ------- GA -------
    t0 = time.time()
    ga_path, ga_cost, _ = genetic_algorithm(dist_matrix)
    t1 = time.time()
    row["GA_cost"] = ga_cost
    row["GA_time"] = round(t1 - t0, 4)

    print(f"GA   : cost={ga_cost:.2f}, time={row['GA_time']} sec")

    # ------- QGA -------
    t0 = time.time()
    qga_path, qga_cost, _ = quantum_genetic_algorithm(dist_matrix)
    t1 = time.time()
    row["QGA_cost"] = qga_cost
    row["QGA_time"] = round(t1 - t0, 4)

    print(f"QGA  : cost={qga_cost:.2f}, time={row['QGA_time']} sec")

    # ------- ACO -------
    t0 = time.time()
    aco_path, aco_cost, _ = ant_colony_optimization(dist_matrix)
    t1 = time.time()
    row["ACO_cost"] = aco_cost
    row["ACO_time"] = round(t1 - t0, 4)

    print(f"ACO  : cost={aco_cost:.2f}, time={row['ACO_time']} sec")

    # ------- Exact solver for small n -------
    if n <= 10:
        t0 = time.time()
        exact_cost, exact_path = solve_exact_tsp(dist_matrix)
        t1 = time.time()
        row["Exact_cost"] = exact_cost
        row["Exact_time"] = round(t1 - t0, 4)
        print(f"EXACT: cost={exact_cost:.2f}, time={row['Exact_time']} sec")
    else:
        row["Exact_cost"] = None
        row["Exact_time"] = None
        print("EXACT: skipped (n > 10)")

    results.append(row)


# =============================
#  Make into table
# =============================
df = pd.DataFrame(results)
print("\n\n============================")
print("📊 Final Results Table")
print("============================")
print(df)

df.to_csv("tsp_experiment_results.csv", index=False)
print("\nResults saved to tsp_experiment_results.csv")
