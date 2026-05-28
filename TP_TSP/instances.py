import numpy as np
import matplotlib.pyplot as plt

def generate_cities(n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    coords = np.random.rand(n, 2) * 100
    return coords

def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d

    return dist_matrix

def plot_cities(coords):
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c='blue')

    for i, (xi, yi) in enumerate(coords):
        plt.text(xi, yi, str(i), fontsize=12)

    plt.title("TSP - Cities")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def plot_tsp_path(coords, path, title="TSP Path"):
    path = list(path)
    path.append(path[0])

    x = [coords[i][0] for i in path]
    y = [coords[i][1] for i in path]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, '-o')

    for i in range(len(path)-1):
        plt.text(coords[path[i]][0], coords[path[i]][1],
                 str(path[i]), fontsize=12)
        
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()