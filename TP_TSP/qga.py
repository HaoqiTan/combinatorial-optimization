import numpy as np
import random
import math

# --- Utility function ---
def tsp_cost(path, dist_matrix):
    cost = 0
    for i in range(len(path)):
        cost += dist_matrix[path[i]][path[(i+1) % len(path)]]
    return cost


# --- Encode the path into binary form ---
def int_to_bin(x, bits):
    return np.array(list(np.binary_repr(x, bits))).astype(int)

def bin_to_int(b):
    return int("".join(str(x) for x in b), 2)


def encode_path(path, bits):
    return np.concatenate([int_to_bin(city, bits) for city in path])

def decode_bitstring(bitstring, n_cities, bits):
    cities = []
    for i in range(n_cities):
        segment = bitstring[i*bits:(i+1)*bits]
        city = bin_to_int(segment) % n_cities
        cities.append(city)
    # Fix duplicates and missing elements (avoid invalid permutations)
    cities = repair_path(cities)
    return cities


# --- Repair invalid paths (maintain GA/QGA feasibility) ---
def repair_path(path):
    n = len(path)
    seen = set()
    result = []
    for c in path:
        if c not in seen and c < n:
            result.append(c)
            seen.add(c)

    # Add missing cities
    for c in range(n):
        if c not in seen:
            result.append(c)
    return result


# --- Q-bit initialization---
def initialize_qbits(length):
    # α = β = 1/√2 → 50/50 state
    return np.ones((length, 2)) / math.sqrt(2)


# --- Measure q-bits to obtain a 0/1 string ---
def measure(qbits):
    binary = []
    for alpha, beta in qbits:
        p1 = beta**2
        binary.append(1 if random.random() < p1 else 0)
    return np.array(binary)


# --- Quantum rotation gate update ---
def update_qbits(qbits, best_bits, current_bits, theta=0.03):
    for i in range(len(qbits)):
        if current_bits[i] == best_bits[i]:
            continue
        # If rotation toward best_bits[i] is required
        if best_bits[i] == 1:
            qbits[i][1] += theta
            qbits[i][0] -= theta
        else:
            qbits[i][1] -= theta
            qbits[i][0] += theta

        # Normalization
        norm = math.sqrt(qbits[i][0]**2 + qbits[i][1]**2)
        qbits[i] /= norm
    return qbits


# --- Main QGA algorithm---
def quantum_genetic_algorithm(dist_matrix, generations=200):
    n = len(dist_matrix)
    bits = math.ceil(math.log2(n))
    qlength = n * bits  # Total binary length

    qbits = initialize_qbits(qlength)

    best_solution = None
    best_cost = float("inf")
    history = []

    for g in range(generations):

        # collapse to actual bitstring
        measured_bits = measure(qbits)
        path = decode_bitstring(measured_bits, n, bits)
        cost = tsp_cost(path, dist_matrix)

        if cost < best_cost:
            best_cost = cost
            best_solution = path
            best_bits = measured_bits.copy()

        # Update the quantum state
        qbits = update_qbits(qbits, best_bits, measured_bits)

        history.append(best_cost)

        if g % 20 == 0:
            print(f"Generation {g}, Best cost = {best_cost}")

    return best_solution, best_cost, history
