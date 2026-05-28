# from quantum_ga import QuantumGA
# from knapsack_instances import generate_knapsack_instances

# instances = generate_knapsack_instances()
# qga = QuantumGA(pop_size=30, n_generations=100)

# for i, (values, weights, capacity) in enumerate(instances):
#     best_sol, best_fit = qga.evolve(values, weights, capacity)
#     print(f"Instance {i+1}: Best fitness = {best_fit}")

import time
import matplotlib.pyplot as plt
from classical_ga import ClassicalGA
from quantum_ga import QuantumGA
from knapsack_instances import generate_knapsack_instances
from utils import knapsack_fitness


# This part run the algorithm and record the results
def run_algorithms_on_instances(instances, pop_size=30, n_gen=100, mutation_rate=0.05):
    results = []

    ga = ClassicalGA(pop_size=pop_size, n_gen=n_gen, mutation_rate=mutation_rate)
    qga = QuantumGA(pop_size=pop_size, n_generations=n_gen)

    for i, (values, weights, capacity) in enumerate(instances):
        print(f"\n Instance {i+1} | Capacity={capacity} | Items={len(values)}")

        # Classical GA
        start_ga = time.time()
        best_ga, fit_ga = ga.evolve(values, weights, capacity)
        time_ga = time.time() - start_ga

        # Quantum GA
        start_qga = time.time()
        best_qga, fit_qga = qga.evolve(values, weights, capacity)
        time_qga = time.time() - start_qga

        # Record the results
        results.append({
            "Instance": i + 1,
            "GA_Fitness": fit_ga,
            "GA_Time": round(time_ga, 3),
            "QGA_Fitness": fit_qga,
            "QGA_Time": round(time_qga, 3)
        })

        print(f"   GA  -> Fitness={fit_ga}, Time={time_ga:.2f}s")
        print(f"   QGA -> Fitness={fit_qga}, Time={time_qga:.2f}s")

    return results


# Draw plots
def plot_results(results):
    instances = [r["Instance"] for r in results]
    ga_fit = [r["GA_Fitness"] for r in results]
    qga_fit = [r["QGA_Fitness"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(instances, ga_fit, "o--", label="Classical GA")
    plt.plot(instances, qga_fit, "s-", label="Quantum GA")
    plt.xlabel("Instance")
    plt.ylabel("Best Fitness")
    plt.title("Comparison: GA vs QGA on Knapsack Problem")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main entrance
if __name__ == "__main__":
    instances = generate_knapsack_instances()
    results = run_algorithms_on_instances(instances, pop_size=30, n_gen=100, mutation_rate=0.05)

    print("\n==================== Summary ====================")
    for r in results:
        print(f"Instance {r['Instance']}: "
              f"GA={r['GA_Fitness']} ({r['GA_Time']}s), "
              f"QGA={r['QGA_Fitness']} ({r['QGA_Time']}s)")

    plot_results(results)
