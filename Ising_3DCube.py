import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt
#AutoCorrelation function
def tau(x, max_lag=None, c=5):
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

        # With these 2 sub-lists of x, we compute the covariance of x and itself shifted over by tau sweeps
        cov = np.cov(a, a_shift)[0, 1]
        rho[tau] = cov / var

    tau = 0.5
    for W in range(1, len(rho)):
        tau += rho[W]
        if W >= c * tau:
            return tau, True                   # window closed: estimate is trustworthy
    return tau, False
# Setting up a 3D lattice - a 3D cube
L = 10 #Side length
N=L**3 #Total number of sites
J = 1.0
T = 20 #Temperature
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
spins = np.random.choice([-1, 1], size=N)
# Inital Magnetization
M = np.sum(spins)
# inital Energy
E = -J * np.sum(A * np.outer(spins, spins)) / 2
print("Initial Energy:", E)
print("Initial Magnetization:", M)
#Metropolis Algorithm
@njit
def metropolis(spins, A, energy, J, T, times=200000):
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

        net_spins[t] =(spin_arr.sum())
        net_energy[t] = energy

    return net_energy, net_spins, spin_arr 

E_hist, M_hist, final_spins = metropolis(spins, A, E, J, T, times=200000)
# --- Autocorrelation Time ---
burn = 10000
tau_E, ok_E = tau(E_hist[burn:])
tau_M, ok_M = tau(np.abs(M_hist[burn:]))

print(f"tau_E = {tau_E: .1f} (Converged: {ok_E})")
print(f"tau_M = {tau_M: .1f} (Converged: {ok_M})")

# Check: where did the system end up?
print("E Final:",E_hist[-1], "M Final:", M_hist[-1]/N)

# ---- Creating averages for Plots ----
block = 1000
n_blocks = len(E_hist) // block
E_block = np.mean(E_hist[:n_blocks*block].reshape(n_blocks, block), axis=1)
M_block = np.mean(M_hist[:n_blocks*block].reshape(n_blocks, block), axis=1) / N
block_centres = np.arange(n_blocks) * block + block / 2

E_mean = np.mean(E_hist)          #Overall average Energy
M_mean = np.mean(M_hist) / N      #Overall average Magnetization

# ---- Plot E and M against sweeps ----
sweeps = np.arange(len(E_hist))     # step number -> sweeps

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(sweeps, E_hist)
ax[0].plot(block_centres, E_block, color='red', label='Sweep Average')
ax[0].axhline(E_mean, color='green', linestyle='--', label='Overall Average')
ax[0].set_xlabel("Sweeps")
ax[0].set_ylabel("Energy E")

ax[1].plot(sweeps, M_hist/N)
ax[1].plot(block_centres, M_block, color='red', label='Sweep Average')
ax[1].axhline(M_mean, color='green', linestyle='--', label='Overall Average')
ax[1].set_xlabel("Sweeps")
ax[1].set_ylabel("Magnetisation |M| Per Spin")

ax[0].legend()
ax[1].legend()

fig.suptitle(f"3D cube, L = {L}, T = {T}")
plt.tight_layout()
plt.show()