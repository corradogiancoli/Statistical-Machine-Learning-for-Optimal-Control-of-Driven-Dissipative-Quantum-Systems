import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import (basis, destroy, mesolve, qeye,tensor, SESolver, sigmaz, sigmam, sigmap, fock)
from scipy.signal import find_peaks

matplotlib.use('TkAgg')


# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
#wa = 1.0 * 2 * np.pi  # atom frequency 
g = 5 * 2 * np.pi # coupling strength

# Hilbert space dimension for the cavity
n_cavity = 75

# Operators
sigma_minus = tensor(qeye(n_cavity), sigmap()) # atomic lowering operator
sigma_plus = sigma_minus.dag()   # atomic raising operator
sigma_z = tensor(qeye(n_cavity), sigmaz().dag())     # atomic z operator
a = tensor(destroy(n_cavity), qeye(2))   # cavity annihilation operator

# Times for which the state should be evaluated
times = np.linspace(0, 25, 50000)

tau = wc * times

def create_Hamiltonian(wa):

    # Hamiltonian with rotating wave approximation
    H_theoretical =  a.dag() * a + 0.5 * (wa/wc) * sigma_z +  (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)

    return H_theoretical


wa_values = np.arange(0, wc, 0.5)

# Prepare plot
fig, axes = plt.subplots(nrows=len(wa_values), figsize=(10, 500), sharex=True, sharey=True)
if len(wa_values) == 1:
    axes = [axes]

for idx, wa in enumerate(wa_values):

    H = create_Hamiltonian(wa=wa)

    n_list = np.arange(1,11,1)

    theo_freq = g * np.sqrt(n_list) #theoretical values

    freq_list = []



    for n in n_list:
        psi = tensor(fock(n_cavity, n), basis(2,0)) + tensor(fock(n_cavity, n-1), basis(2,1))
        #print(psi.dims)
        output = mesolve(H, psi, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
        n_c = output.expect[0]

        peaks, _ = find_peaks(n_c, height=0.5) 

        # Calculate time points of peaks
        peak_times = tau[peaks]
        periods = np.diff(peak_times)
        average_period = np.mean(periods)
        frequency = 1 / average_period if average_period != 0 else 0

        freq_list.append(frequency * np.pi * wc)

    # Plot
    ax = axes[idx]
    ax.plot(n_list, theo_freq, label='Theoretical Frequency', marker='o')
    ax.plot(n_list, freq_list, label='Experimental Frequency', linestyle='--', marker='x')
    ax.set_title(f"w_a/w_c = {wa/wc:.2f}")
    ax.set_xlabel("Fock state n")
    ax.set_ylabel("Frequency (rad/s)")
    ax.legend(loc='best')
    ax.grid(True)

plt.tight_layout()
plt.show()
