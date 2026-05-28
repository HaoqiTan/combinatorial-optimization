from ortools.linear_solver import pywraplp
import numpy as np


def solve_exact_tsp(dist_matrix):
    """
    Solve the symmetric TSP exactly using OR-Tools (SCIP solver)
    with Miller-Tucker-Zemlin (MTZ) constraints to eliminate subtours.

    Args:
        dist_matrix (np.array): distance matrix of shape (n, n)

    Returns:
        best_cost (float): optimal tour cost
        best_path (list): visiting order of cities in the optimal tour
    """
    n = len(dist_matrix)
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # --- Decision variables x[i][j]: whether city i goes to j ---
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = solver.IntVar(0, 1, f"x[{i},{j}]")

    # --- MTZ ordering variables t[i] ---
    # Used to prevent subtours (Miller-Tucker-Zemlin formulation)
    t = {}
    for i in range(1, n):
        t[i] = solver.NumVar(1, n - 1, f"t[{i}]")

    # --- Constraints: each city has exactly one outgoing and one incoming edge ---
    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n) if j != i]) == 1)
        solver.Add(solver.Sum([x[j, i] for j in range(n) if j != i]) == 1)

    # --- MTZ subtour elimination constraints ---
    B = n - 1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                solver.Add(t[j] >= t[i] + 1 - B * (1 - x[i, j]))

    # --- Objective: minimize total travel distance ---
    objective_terms = []
    for i in range(n):
        for j in range(n):
            if i != j:
                objective_terms.append(dist_matrix[i][j] * x[i, j])

    solver.Minimize(solver.Sum(objective_terms))

    # --- Solve the MILP problem ---
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        return None, None

    # --- Extract optimal tour ---
    # We reconstruct the path starting from node 0
    path = [0]
    current = 0

    for _ in range(n - 1):
        for j in range(n):
            if j != current and x[current, j].solution_value() > 0.5:
                path.append(j)
                current = j
                break

    # --- Compute final cost of the optimal tour ---
    best_cost = sum(dist_matrix[path[i]][path[(i + 1) % n]] for i in range(n))

    return best_cost, path
