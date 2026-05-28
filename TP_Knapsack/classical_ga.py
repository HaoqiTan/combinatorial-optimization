from utils import knapsack_fitness

class ClassicalGA:
    def __init__(self, pop_size, n_gen, mutation_rate):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate

    def evolve(self, values, weights, capacity):
        import numpy as np
        n = len(values)
        pop = np.random.randint(0, 2, (self.pop_size, n))
        best = None
        best_fit = 0

        for _ in range(self.n_gen):
            fitness = [knapsack_fitness(ind, values, weights, capacity) for ind in pop]
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > best_fit:
                best_fit = fitness[best_idx]
                best = pop[best_idx].copy()

            # Selection
            probs = np.array(fitness) / (sum(fitness) + 1e-9)
            parents = pop[np.random.choice(self.pop_size, size=self.pop_size, p=probs)]

            # Crossover
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[(i+1) % self.pop_size]
                point = np.random.randint(1, n-1)
                child1 = np.concatenate([p1[:point], p2[point:]])
                child2 = np.concatenate([p2[:point], p1[point:]])
                offspring.extend([child1, child2])

            # Mutation
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
            offspring[mutation_mask] ^= 1

            pop = offspring

        return best, best_fit