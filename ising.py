import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

N = 4 # grid size
J = 1 # Exchange energy
Kb = 1 # Boltzmann constant
T = 2 # temperature

def get_energy(lattice):
    kern = generate_binary_structure(2, 1) 
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    return 1/2 * arr.sum()

@njit
def metropolis(spin_matrix, energy, times=100000, KbT=Kb*T):
    # Initialize spin and energy lists
    spin_arr = spin_matrix.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)

    for t in range(times-1):
        # Randomly choose a cell in spin matrix and flip it
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        spin_i = spin_arr[i][j]
        spin_f = -1 * spin_arr[i][j]

        # Compute change in energy from proposed flip
        Ei, Ef = 0, 0
        if j < N-1:
            Ei += -spin_i * spin_arr[i][j+1]
            Ef += -spin_f * spin_arr[i][j+1]
        if j > 0:
            Ei += -spin_i * spin_arr[i][j-1]
            Ef += -spin_f * spin_arr[i][j-1]
        if i < N-1:
            Ei += -spin_i * spin_arr[i+1][j]
            Ef += -spin_f * spin_arr[i+1][j]
        if i > 0:
            Ei += -spin_i * spin_arr[i-1][j]
            Ef += -spin_f * spin_arr[i-1][j]

        dE = Ef - Ei

        # Accept proposed state based on probability given by min(1, exp(-dE/Kb*T))
        if dE <= 0:
            spin_arr[i][j] = spin_f
            energy += dE
        elif dE > 0:
            probability = np.exp(-dE / KbT)
            random_n = np.random.random()
            if random_n <= probability:
                spin_arr[i][j] = spin_f
                energy += dE

        net_spins[t] = np.abs(spin_arr.sum())
        net_energy[t] = energy


    return net_energy, net_spins # E, M     


def main():
    random_matrix = np.random.random((N,N))
    spin_matrix = np.zeros((N, N))
    spin_matrix[random_matrix >= 0.5] = 1
    spin_matrix[random_matrix < 0.5] = -1

    initial_energy = get_energy(spin_matrix)
    E, M = metropolis(spin_matrix, initial_energy)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(E); ax[0].set_ylabel('Energy'); ax[0].set_xlabel('Sweep')
    ax[1].plot(M / N**2); ax[1].set_ylabel('Magnetization per spin'); ax[1].set_xlabel('Sweep')
    plt.subplots_adjust(wspace=0.6, hspace=0.6) 

    # plot average every 10-1000 sweeps

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()