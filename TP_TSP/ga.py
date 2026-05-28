import random
import numpy as np
import matplotlib.pyplot as plt

# --- Basic functions ---
def tsp_cost(path, dist_matrix):
    cost = 0
    n = len(path)
    for i in range(n):
        cost += dist_matrix[path[i]][path[(i + 1) % n]]
    return cost


# --- Initialization ---
def init_population(pop_size, n_cities):
    population = []
    for _ in range(pop_size):
        path = list(range(n_cities))
        random.shuffle(path)
        population.append(path)
    return population


# --- Selection: tournament selection ---
def tournament_selection(population, dist_matrix, k=3):
    candidates = random.sample(population, k)
    candidates_costs = [tsp_cost(c, dist_matrix) for c in candidates]
    best = candidates[np.argmin(candidates_costs)]
    return best


# --- OX crossover (Order Crossover) ---
def order_crossover(parent1, parent2):
    n = len(parent1)
    child = [None] * n

    a, b = sorted(random.sample(range(n), 2))

    # copy segment from parent1
    child[a:b] = parent1[a:b]

    # fill the remaining from parent2
    fill_pos = b
    for city in parent2:
        if city not in child:
            if fill_pos >= n:
                fill_pos = 0
            child[fill_pos] = city
            fill_pos += 1

    return child


# --- Mutation ---
def mutate(path, mutation_rate=0.1):
    new_path = path.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(path)), 2)
        new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


# --- Main GA loop ---
def genetic_algorithm(dist_matrix, pop_size=100, generations=500, mutation_rate=0.1):
    n_cities = len(dist_matrix)
    population = init_population(pop_size, n_cities)
    
    best_costs = []
    
    # Find the global best
    best_individual = min(population, key=lambda p: tsp_cost(p, dist_matrix))
    best_cost = tsp_cost(best_individual, dist_matrix)

    for g in range(generations):

        new_population = []

        for _ in range(pop_size):
            parent1 = tournament_selection(population, dist_matrix)
            parent2 = tournament_selection(population, dist_matrix)
            child = order_crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Update the best solution
        current_best = min(population, key=lambda p: tsp_cost(p, dist_matrix))
        current_cost = tsp_cost(current_best, dist_matrix)

        if current_cost < best_cost:
            best_cost = current_cost
            best_individual = current_best

        best_costs.append(best_cost)

        if g % 50 == 0:
            print(f"Generation {g}, Best Cost = {best_cost}")

    return best_individual, best_cost, best_costs
