import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

N = 4  # grid size
J = 1  # Exchange energy
Kb = 1  # Boltzmann constant
T = 2  # temperature


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

    return net_energy, net_spins  # E, M


def wolff(spin_matrix, energy, times=100000, KbT=Kb*T):
    # Initialize spin and energy lists
    spin_arr = spin_matrix.copy()
    net_spins = np.zeros(times-1)
    net_energy = np.zeros(times-1)
    energy = get_energy(spin_arr)

    # probability of addition
    p_add = 1 - np.exp(-2 * J / KbT)

    for t in range(times - 1):
        # pick random seed spin to begin cluster
        i0 = np.random.randint(0, N)
        j0 = np.random.randint(0, N)
        seed_spin = spin_arr[i0, j0]
        seed_spin = spin_arr[i0][j0]

        # assign start of stack to the seeds
        stack = [(i0, j0)]
        # immediately flip the seed
        spin_arr[i0][j0] = -seed_spin

        while len(stack) > 0:
            curr_i, curr_j = stack.pop()

            neighbors = []
            if curr_j < N - 1:
                neighbors.append((curr_i, curr_j + 1))
            if curr_j > 0:
                neighbors.append((curr_i, curr_j - 1))
            if curr_i < N - 1:
                neighbors.append((curr_i + 1, curr_j))
            if curr_i > 0:
                neighbors.append((curr_i - 1, curr_j))

            # loop through neighbors to flip
            for ni, nj in neighbors:
                if spin_arr[ni][nj] == seed_spin:
                    if np.random.random() < p_add:
                        spin_arr[ni][nj] = -seed_spin
                        stack.append((ni, nj))

        # recompute after loop
        energy = get_energy(spin_arr)
        net_spins[t] = np.abs(spin_arr.sum())
        net_energy[t] = energy

    return net_energy, net_spins


def main():
    random_matrix = np.random.random((N, N))
    spin_matrix = np.zeros((N, N))
    spin_matrix[random_matrix >= 0.5] = 1
    spin_matrix[random_matrix < 0.5] = -1

    initial_energy = get_energy(spin_matrix)
    E_metro, M_metro = metropolis(spin_matrix.copy(), initial_energy)
    E_wolff, M_wolff = wolff(spin_matrix.copy(), initial_energy)

    # clean array generation ensuring identical dimensions
    steps = np.arange(1000, len(E_metro) + 1, 1000)
    energies_m = [np.average(E_metro[i-100:i]) for i in steps]
    mag_m = [np.average(M_metro[i-100:i]) for i in steps]
    energies_w = [np.average(E_wolff[i-100:i]) for i in steps]
    mag_w = [np.average(M_wolff[i-100:i]) for i in steps]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(steps, np.array(energies_m) / N**2, label='Metropolis')
    ax[0].plot(steps, np.array(energies_w) / N**2, label='Wolff')
    ax[0].set_ylabel('Energy per spin')
    ax[0].set_xlabel('Sweeps')
    ax[0].legend()

    ax[1].plot(steps, np.array(mag_m) / N**2, label='Metropolis')
    ax[1].plot(steps, np.array(mag_w) / N**2, label='Wolff')
    ax[1].set_ylabel('Magnetization per spin')
    ax[1].set_xlabel('Sweeps')

    plt.subplots_adjust(wspace=0.6)
    plt.show()


if __name__ == "__main__":
    main()
