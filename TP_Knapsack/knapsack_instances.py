import numpy as np

def generate_knapsack_instances(num_instances=5, n_items=20, seed=42):
    np.random.seed(seed)
    instances = []

    for i in range(num_instances):
        # randomly generate values and weights
        values = np.random.randint(10, 100, size=n_items)
        weights = np.random.randint(5, 50, size=n_items)

        # Set the capacity to about one-third of the total weight
        capacity = int(sum(weights) / 3)

        # To ensure instance diversity, make some instances value-oriented and others weight-oriented.
        if i % 2 == 0:
            values = np.sort(values)[::-1]       # This type prioritize the value
        else:
            weights = np.sort(weights)           # This type prioritize to reduce the weight

        instances.append((values, weights, capacity))

    return instances


if __name__ == "__main__":
    instances = generate_knapsack_instances()
    for i, (v, w, c) in enumerate(instances):
        print(f"Instance {i+1}: capacity={c}")
        print(f"  values : {v[:10]} ...")
        print(f"  weights: {w[:10]} ...")
        print("-" * 50)
