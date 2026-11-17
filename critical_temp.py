from ising import metropolis, tau, get_energy
import numpy as np
import matplotlib.pyplot as plt

N = 10
Kb = 1

def error(x, t_int):
    n_eff = len(x) / (2 * t_int)
    return np.std(x, ddof=1) / np.sqrt(n_eff)

def main():
    temps = np.concatenate([
        np.arange(1.0, 2.0, 0.25),
        np.arange(2.0, 3.0, 0.1),
        np.arange(3.0, 5.01, 0.5),
    ])

    m_avg, m_err = [], []
    e_avg, e_err = [], []
    unconverged = []

    n_steps = 100000
    for temp in temps:
        spins = np.ones((N, N))                  # ordered start
        E0 = get_energy(spins)

        E, M = metropolis(spins, E0, times=n_steps, KbT=Kb * temp)

        burn = len(E) // 5
        E_eq = E[burn:]
        absM_eq = np.abs(M[burn:])

        tau_E, ok_E = tau(E_eq)
        tau_M, ok_M = tau(absM_eq)
        if not (ok_E and ok_M):
            unconverged.append(temp)

        m_avg.append(absM_eq.mean() / N**2)
        m_err.append(error(absM_eq, tau_M) / N**2)
        e_avg.append(E_eq.mean() / N**2)
        e_err.append(error(E_eq, tau_E) / N**2)


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(temps, m_avg, yerr=m_err, fmt="o-")
    ax.set_ylabel("<M> per spin")
    ax.set_xlabel("T")
    ax.axvline(2.269, color="k", ls="--", lw=0.8)   # infinite-lattice T_c
    plt.tight_layout()
    plt.savefig(f'figures/critical_temp')
    plt.show()

    if unconverged:
        print("tau window did not close at T =", unconverged, "(error bars there are underestimates)")

if __name__ == "__main__":
    main()