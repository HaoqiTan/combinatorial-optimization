import numpy as np
import random
import math

def tsp_cost(path, dist_matrix):
    cost = 0
    n = len(path)
    for i in range(n):
        cost += dist_matrix[path[i]][path[(i+1)%n]]
    return cost


def ant_colony_optimization(dist_matrix, n_ants=20, generations=200, 
                            alpha=1.0, beta=5.0, evaporation=0.5, Q=100):
    n = len(dist_matrix)
    
    # 1. pheromone initialization
    pheromone = np.ones((n, n))
    
    # 2. Heuristic information (inverse distance)
    heuristic = 1 / (dist_matrix + 1e-10)
    
    best_cost = float("inf")
    best_path = None
    history = []
    
    for gen in range(generations):
        all_paths = []
        all_costs = []
        
        # --- Each ant constructs a path ---
        for ant in range(n_ants):
            
            unvisited = list(range(n))
            start = random.choice(unvisited)
            path = [start]
            unvisited.remove(start)
            
            # Construct the path according to probabilistic rules
            while unvisited:
                i = path[-1]
                
                # Compute the probabilities
                probabilities = []
                for j in unvisited:
                    probabilities.append(
                        (pheromone[i][j] ** alpha) * (heuristic[i][j] ** beta)
                    )
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                
                # Select the next city according to the probabilities
                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                unvisited.remove(next_city)
            
            cost = tsp_cost(path, dist_matrix)
            all_paths.append(path)
            all_costs.append(cost)
        
        # Update the best solution
        gen_best_cost = min(all_costs)
        gen_best_path = all_paths[np.argmin(all_costs)]
        
        if gen_best_cost < best_cost:
            best_cost = gen_best_cost
            best_path = gen_best_path
        
        history.append(best_cost)
        
        # --- Pheromone update: evaporation ---
        pheromone *= (1 - evaporation)
        
        # --- Pheromone reinforcement (shorter paths contribute more) ---
        for path, cost in zip(all_paths, all_costs):
            deposit = Q / cost
            for i in range(n):
                a = path[i]
                b = path[(i+1)%n]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
        
        if gen % 20 == 0:
            print(f"Generation {gen}, Best Cost = {best_cost}")

    return best_path, best_cost, history


