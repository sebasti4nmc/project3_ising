import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

N = 100 # grid size
J = 1 # Exchange energy
Kb = 1 # Boltzmann constant
T = 1 # temperature

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

def autocorr(x, max_lag=None):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 10

    mu = np.mean(x)
    var = np.var(x)    

    rho = np.empty(max_lag)
    for tau in range(max_lag):
        a = x[:n - tau]                 # Compute x_t, our quantity x up to sweep n-tau
        a_shift = x[tau:]               # Compute x_{t+tau}, our quantity x shifted by value tau

        # With these 2 sublists of x, we compute the covariance of x and itself shifted over by tau sweeps
        cov = np.cov(a, a_shift)[0, 1]
        rho[tau] = cov / var

    return 0.5 + rho.sum()


def main():
    random_matrix = np.random.random((N,N))
    spin_matrix = np.zeros((N, N))
    spin_matrix[random_matrix >= 0.5] = 1
    spin_matrix[random_matrix < 0.5] = -1

    initial_energy = get_energy(spin_matrix)
    E, M = metropolis(spin_matrix, initial_energy)

    tau_E = autocorr(E)
    tau_M = autocorr(M)

    print(tau_M)

    steps = []
    energies = []
    mag = []

    for i in range(1000, len(E) + 1, 1000):
        energies.append(np.average(E[i-100:i]))
        mag.append(np.average(M[i-100:i]))
        steps.append(i)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(steps, np.array(energies))
    ax[0].set_ylabel('Energy')
    ax[0].set_xlabel('Sweeps')

    ax[1].plot(steps, np.array(mag) / N**2)
    ax[1].set_ylabel('Magnetization per spin')
    ax[1].set_xlabel('Sweeps')

    plt.subplots_adjust(wspace=0.6)
    plt.show()

if __name__ == "__main__":
    main()