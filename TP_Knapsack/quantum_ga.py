import numpy as np
from utils import knapsack_fitness

class QuantumGA:
    def __init__(self, pop_size=30, n_generations=100, theta=0.05 * np.pi):

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.theta = theta

    # Initialize
    def initialize_population(self, n_bits):
        alpha = np.ones((self.pop_size, n_bits)) / np.sqrt(2)
        beta = np.ones((self.pop_size, n_bits)) / np.sqrt(2)
        return alpha, beta

    # observe the quantum state
    def observe(self, alpha, beta):
        n_ind, n_bits = alpha.shape
        population = np.zeros((n_ind, n_bits), dtype=int)
        for i in range(n_ind):
            probs = np.abs(beta[i]) ** 2  # the probability of "1"
            population[i] = (np.random.rand(n_bits) < probs).astype(int)
        return population

    # update of the rotation gate
    def rotation_gate(self, alpha, beta, bit, best_bit, fit_better):
        delta = 0.0
        sign = 0

        if bit == 0 and best_bit == 1:
            delta = 0.05 * np.pi if fit_better else 0.025 * np.pi
            sign = +1
        elif bit == 1 and best_bit == 0:
            delta = 0.01 * np.pi if fit_better else 0.025 * np.pi
            sign = -1
        elif bit == 1 and best_bit == 1:
            delta = 0.05 * np.pi if fit_better else 0.025 * np.pi
            sign = -1
        elif bit == 0 and best_bit == 0:
            delta = 0

        # rotation matrix
        rotation_matrix = np.array([
            [np.cos(sign * delta), -np.sin(sign * delta)],
            [np.sin(sign * delta), np.cos(sign * delta)]
        ])

        vec = np.array([alpha, beta])
        new_vec = rotation_matrix @ vec
        return new_vec[0], new_vec[1]

    # Main process
    def evolve(self, values, weights, capacity):
        n_bits = len(values)
        alpha, beta = self.initialize_population(n_bits)
        best_solution = None
        best_fitness = -1

        for gen in range(self.n_generations):
            # observe and assess
            population = self.observe(alpha, beta)
            fitnesses = [knapsack_fitness(ind, values, weights, capacity) for ind in population]

            # find the optimal answer
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fit = fitnesses[gen_best_idx]
            gen_best_sol = population[gen_best_idx]

            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_solution = gen_best_sol.copy()

            # renew qubits
            for j in range(self.pop_size):
                fit_better = fitnesses[j] < best_fitness
                for i in range(n_bits):
                    alpha[j, i], beta[j, i] = self.rotation_gate(
                        alpha[j, i], beta[j, i],
                        population[j, i],
                        best_solution[i],
                        fit_better
                    )

        return best_solution, best_fitness
