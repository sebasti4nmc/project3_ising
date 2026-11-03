import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
# Setting up a 3D lattice - a 3D cube
L = 3 #Side length
N=L**3 #Total number of sites
J = 1.0
T = 2.0 #Temperature
#Coordinate system for the 3D lattice
def site_number(x, y, z):
    return x * L**2 + y*L + z
# Setting up Adjacency Matrix
A = np.zeros((N, N), dtype=int)
for x in range(L):
    for y in range(L):
        for z in range(L):
            i = site_number(x, y, z)

            A[i, site_number((x + 1) % L, y, z)] = 1
            A[i, site_number((x - 1) % L, y, z)] = 1
            A[i, site_number(x, (y + 1) % L, z)] = 1
            A[i, site_number(x, (y - 1) % L, z)] = 1
            A[i, site_number(x, y, (z + 1) % L)] = 1
            A[i, site_number(x, y, (z - 1) % L)] = 1

# Setting up the initial spin configuration - Hot Start
np.random.seed(42)
spins = np.random.choice([-1, 1], size=N)
# Inital Magnetization
M = np.sum(spins)
# inital Energy
E = -J * np.sum(A * np.outer(spins, spins)) / 2
print(A.sum(axis=1)) # This will print the number of neighbors for each site
print(np.array_equal(A, A.T))
print(np.nonzero(A[13])[0])

test = np.ones(N)
print(-J * np.sum(A * np.outer(test, test)) / 2) # This should be -3*L^3
print(E, M)

#Metropolis Algorithm
@njit
def metropolis(spins, A, energy, J, T, times=10000):
    spin_arr = spins.copy()
    N = spin_arr.size
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)

    for t in range(times - 1):
        i = np.random.randint(0, N)
        spin_i = spin_arr[i]
        spin_f = -1 *spin_arr[i]

        neighbour_sum = 0
        for j in range(N):
            if A[i, j] == 1:
                neighbour_sum += spin_arr[j]

        Ei = -J * spin_i * neighbour_sum
        Ef = -J * spin_f * neighbour_sum
        dE= Ef - Ei

        if dE <= 0:
            spin_arr[i] = spin_f
            energy += dE
        else:
            probability = np.exp(-dE / T)
            if np.random.random() <= probability:
                spin_arr[i] = spin_f
                energy += dE

        net_spins[t] = np.abs(spin_arr.sum())
        net_energy[t] = energy

    return net_energy, net_spins, spin_arr 

E_hist, M_hist, final_spins = metropolis(spins, A, E, J, T, times=10000)

# Check 1: running energy should match the Hamiltonian recomputed from scratch
E_recomputed = -J * np.sum(A * np.outer(final_spins, final_spins)) / 2
print(E_hist[-1], E_recomputed)          # these two should be equal

# Check 2: where did the system end up?
print(E_hist[-1] / N, M_hist[-1] / N)