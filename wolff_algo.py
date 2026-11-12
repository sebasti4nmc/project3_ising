import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure

N = 10  # grid size
J = 1  # Exchange energy
Kb = 1  # Boltzmann constant
T = 20  # temperature


def get_energy(lattice):
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern, mode='constant', cval=0)
    # print(type(arr.sum()))
    return float(1/2 * arr.sum())


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
    # print(type(spin_arr))
    net_spins = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    cluster_sizes = np.zeros(times - 1)
    energy = get_energy(spin_arr)

    # probability that bond forms between 2 parallel neighboring spins
    p_add = 1 - np.exp(-2 * J / KbT)

    for t in range(times - 1):
        # pick random seed spin to begin cluster
        i0 = np.random.randint(0, N)
        j0 = np.random.randint(0, N)
        seed_spin = spin_arr[i0, j0]
        seed_spin = spin_arr[i0][j0]

        # create a stacking list to track cluster growth
        stack = [(i0, j0)]
        # immediately flip seed
        spin_arr[i0][j0] = -seed_spin

        while len(stack) > 0:
            # pop a coordinate off the stack to check its 4 neighbors
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

            # check if neighbor is in same direction as original seed spin
            for ni, nj in neighbors:
                if spin_arr[ni][nj] == seed_spin:
                    # in less than p_add as well, add it to the cluster
                    if np.random.random() < p_add:
                        spin_arr[ni][nj] = -seed_spin
                        stack.append((ni, nj))

        if (t + 1) % 10000 == 0:
            print(f'Wolff sweep {t + 1}')

        # once the cluster stops growing, compute total energy and magnitization
        energy = get_energy(spin_arr)
        net_spins[t] = np.abs(spin_arr.sum())
        net_energy[t] = energy

    return net_energy, net_spins

# --- AUTOCORRELATION ---


def tau_int(x, max_lag=None, c=5):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 10

    var = np.var(x)
    rho = np.empty(max_lag)
    for tau in range(max_lag):
        a = x[:n - tau]
        a_shift = x[tau:]
        cov = np.cov(a, a_shift)[0, 1]
        rho[tau] = cov / var

    tau = 0.5
    for W in range(1, len(rho)):
        tau += rho[W]
        if W >= c * tau:
            return tau, True
    return tau, False


def main():
    random_matrix = np.random.random((N, N))
    spin_matrix = np.zeros((N, N))
    spin_matrix[random_matrix >= 0.5] = 1
    spin_matrix[random_matrix < 0.5] = -1

    E0 = get_energy(spin_matrix)
    E_metro, M_metro = metropolis(spin_matrix.copy(), E0)
    E_wolff, M_wolff = wolff(spin_matrix.copy(), E0)

    # absolute values
    absM_metro = np.abs(M_metro)
    absM_wolff = np.abs(M_wolff)

    # wolff burn in less than metro burn in
    burn_metro = 4000
    burn_wolff = 1000

    # calculate tau for both methods --> 1 index = 1 sweep
    tau_E_m, ok_E_m = tau_int(E_metro[burn_metro:])
    tau_M_m, ok_M_m = tau_int(absM_metro[burn_metro:])
    tau_E_w, ok_E_w = tau_int(E_wolff[burn_wolff:])
    tau_M_w, ok_M_w = tau_int(absM_wolff[burn_wolff:])

    print(f"\nMetropolis Results")
    print(f"tau_E = {tau_E_m:.2f} sweeps (converged: {ok_E_m})")
    print(f"tau_M = {tau_M_m:.2f} sweeps (converged: {ok_M_m})")
    print(f"\nWolff Results")
    print(f"tau_E = {tau_E_w:.2f} sweeps (converged: {ok_E_w})")
    print(f"tau_M = {tau_M_w:.2f} sweeps (converged: {ok_M_w})")

    # clean array generation ensuring identical dimensions
    steps = np.arange(100, len(E_metro) + 1, 100)
    energies_m = [np.average(E_metro[i-100:i]) for i in steps]
    mag_m = [np.average(absM_metro[i-100:i]) for i in steps]
    energies_w = [np.average(E_wolff[i-100:i]) for i in steps]
    mag_w = [np.average(absM_wolff[i-100:i]) for i in steps]

    # compute averages after burn in
    E_avg_m = E_metro[burn_metro:].mean() / N**2
    E_avg_w = E_wolff[burn_wolff:].mean() / N**2
    M_avg_m = absM_metro[burn_metro:].mean() / N**2
    M_avg_w = absM_wolff[burn_wolff:].mean() / N**2

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(steps, np.array(energies_m) / N**2, label='Metropolis')
    ax[0].plot(steps, np.array(energies_w) / N**2, label='Wolff')
    ax[0].set_ylabel('Energy per spin')
    ax[0].set_xlabel('Sweeps')
    ax[0].axvline(burn_metro, color="k", ls="--", lw=0.8)  # burn in cutoff
    # add in tau to display
    ax[0].plot([], [], " ", label=rf"Metro: $\tau_E$ = {tau_E_m:.1f} sweeps")
    ax[0].plot([], [], " ", label=rf"Wolff: $\tau_E$ = {tau_E_w:.1f} sweeps")
    ax[0].axhline(E_avg_m, color='C0', ls='--', lw=1.2,
                  label=f"Metro <E>/N² = {E_avg_m:.3f}")
    ax[0].axhline(E_avg_w, color='C1', ls='--', lw=1.2,
                  label=f"Wolff <E>/N² = {E_avg_w:.3f}")
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fontsize=8, frameon=False)

    ax[1].plot(steps, np.array(mag_m) / N**2, label='Metropolis')
    ax[1].plot(steps, np.array(mag_w) / N**2, label='Wolff')
    ax[1].set_ylabel('Magnetization per spin')
    ax[1].set_xlabel('Sweeps')
    ax[1].axvline(burn_metro, color="k", ls="--", lw=0.8)
    # add tau into display
    ax[1].plot([], [], " ", label=rf"Metro: $\tau_M$ = {tau_M_m:.1f} sweeps")
    ax[1].plot([], [], " ", label=rf"Wolff: $\tau_M$ = {tau_M_w:.1f} sweeps")
    ax[1].axhline(M_avg_m, color='C0', ls='--', lw=1.2,
                  label=f"Metro <|M|>/N² = {M_avg_m:.3f}")
    ax[1].axhline(M_avg_w, color='C1', ls='--', lw=1.2,
                  label=f"Wolff <|M|>/N² = {M_avg_w:.3f}")
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                 fontsize=8, frameon=False)

    fig.suptitle(f"Metropolis vs. Wolff Algorithm for T = {T}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f'wolff_plots_T{T}.png', bbox_inches='tight', dpi=200)
    print(f'Done plotting wolff_plots_T{T}.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
