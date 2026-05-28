def knapsack_fitness(solution, values, weights, capacity):
    total_value = sum(v for v, bit in zip(values, solution) if bit)
    total_weight = sum(w for w, bit in zip(weights, solution) if bit)
    if total_weight > capacity:
        return 0  # violate the constraint
    return total_value