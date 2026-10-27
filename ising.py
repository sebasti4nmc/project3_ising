import numpy as np
# import random 
import matplotlib.pyplot as plt

N = 4 # grid size
J = 1 # energy per spin^2 (??)
K_b = 1 # Boltzmann constant

def get_energy(lattice):
    energy = 0
    for i in range(len(lattice)):
        for j in range(len(lattice[i])):
            cell = lattice[i][j]
            if j < N-1:
                energy += cell * lattice[i][j+1]
            if i < N-1:
                energy += cell * lattice[i+1][j]
    return -1 * energy

def metropolis(spin_matrix, energy, times=10, KbT=1):
    # Randomly choose a cell in spin matrix and flip it
    i = np.random.randint(0, N)
    j = np.random.randint(0, N)
    spin_i = spin_matrix[i][j]
    spin_f = -1 * spin_matrix[i][j]

    # Compute change in energy from proposed flip
    Ei, Ef = 0, 0
    if j < N-1:
       Ei += -spin_i * spin_matrix[i][j+1]
       Ef += -spin_f * spin_matrix[i][j+1]
    if j > 0:
       Ei += -spin_i * spin_matrix[i][j-1]
       Ef += -spin_f * spin_matrix[i][j-1]
    if i < N-1:
       Ei += -spin_i * spin_matrix[i+1][j]
       Ef += -spin_f * spin_matrix[i+1][j]
    if i > 0:
        Ei += -spin_i * spin_matrix[i-1][j]
        Ef += -spin_f * spin_matrix[i-1][j]

    dE = Ef - Ei

    # Accept proposed state based on probability given by min(1, exp(-dE/Kb*T))
    if dE <= 0:
        spin_matrix[i][j] = spin_f
        energy += dE
    elif dE > 0:
        probability = np.exp(-dE / KbT)
        random_n = np.random.random()
        if random_n <= probability:
            spin_matrix[i][j] = spin_f
            energy += dE

    return energy, spin_matrix.sum() # E, M
            


def main():
    random_matrix = np.random.random((N,N))
    spin_matrix = np.zeros((N, N))
    spin_matrix[random_matrix >= 0.5] = 1
    spin_matrix[random_matrix < 0.5] = -1

    initial_energy = get_energy(spin_matrix)
    E, M = metropolis(spin_matrix, initial_energy)

    print(initial_energy, E, M)


main()

if __name__ == "__main__":
    main()