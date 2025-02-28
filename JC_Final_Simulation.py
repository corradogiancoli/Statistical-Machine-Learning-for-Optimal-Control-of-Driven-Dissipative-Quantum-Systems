import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import (basis, destroy, mesolve, qeye, tensor, create, sigmaz, sigmam, sigmap, fock, Qobj)
from scipy.signal import find_peaks

matplotlib.use('TkAgg')

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.25 * 2 * np.pi # coupling strength

# Hilbert space dimension for the cavity
n_cavity = 15

sigma_z_np = np.array([[1 , 0],
                   [0, -1]])
sigma_z_alt = Qobj(sigma_z_np)

sigma_plus_np = np.array([[0, 1],
                   [0, 0]])

sigma_plus_alt = Qobj(sigma_plus_np)

sigma_minus_np = np.transpose(sigma_plus_np)

sigma_minus_alt = Qobj(sigma_minus_np)

def create_a(n):

    matrix = np.zeros((n, n))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j == i - 1:
                matrix[i , j] = np.sqrt(i)

    return Qobj(np.transpose(matrix))


###########################
# Define the Operators
###########################

sigma_minus = tensor(qeye(n_cavity), sigma_minus_alt) # atomic lowering operator
sigma_plus =  tensor(qeye(n_cavity), sigma_plus_alt)  # atomic raising operator
sigma_z = tensor(qeye(n_cavity), sigma_z_alt)     # atomic z operator
a = tensor(create_a(n_cavity), qeye(2))   # cavity annihilation operator
spin_down = fock(2, 1)
spin_up = fock(2, 0)

##########################
# Run Simulation
##########################

# Times for which the state should be evaluated
times = np.linspace(0, 10, 100000)

tau = wc * times

y = lambda n, tau: np.sin(((g/wc) * np.sqrt(n) * tau))**2

# Hamiltonian with rotating wave approximation
H_theoretical =  a.dag() * a + 0.5 * (wa/wc) * sigma_z +  (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)


# Initial state: superposition of state
psi0 = tensor(fock(n_cavity, 1), spin_up)

# Solve the Schrodinger equation
result = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
n_c = result.expect[0]
n_a = result.expect[1]

##########################
# Visualise Simulation
##########################

n_list = np.arange(1,5,1)

theo_freq = g * np.sqrt(n_list)#theoretical values

freq_list = []
alt_freq_list = []
error_list = []


for n in n_list:
    
    psi1 = tensor(fock(n_cavity, n), spin_down)
    output1 = mesolve(H_theoretical, psi1, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
    n_c = output1.expect[0] 
    n_a = output1.expect[1] 

    y_vals = y(n, tau)

    peaks, _ = find_peaks(n_c, height=0.5)



    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(tau, n_c, label="Cavity Photon Number")
    axes.plot(tau, n_a, label="Atom Excitation")
    axes.plot(tau, y_vals, label=r'Theoretical Dynamics ($\sin(g \: \sqrt{n} \: t)$)', linestyle = 'dashed')
    axes.scatter(tau[peaks], np.array(n_a)[peaks], color='red', s=10, marker='o', label='Peaks')
    

    axes.grid(True)
    axes.legend(loc='best')
    axes.set_xlabel("Time (arb.units)")
    axes.set_ylabel("Photon number")
    axes.set_title(f"Photon and Atom Excitation Dynamics for n={n}")
    plt.savefig(f'JC_plot_for_n:{n}')
    plt.show()



    
    peak_times = times[peaks]
    periods = np.diff(peak_times)
    average_period = np.mean(periods)
    error_period = np.std(periods)
    frequency = 1 / average_period
    error_frequency = error_period / average_period**2

    error_list.append(error_frequency * np.pi)
    freq_list.append(frequency * np.pi)



plt.figure()
plt.errorbar(n_list, freq_list, marker = 'x', color = 'blue', yerr=error_list, fmt = 'o', capsize=4, label='Frequency with error bars')
plt.scatter(n_list, theo_freq, color = 'red', marker = 'x', label = 'theoretical prediction')
plt.xlabel('Fock state of cavity')
plt.ylabel('Frequency (arb. units)')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('Rabi_oscillations_final.png')
plt.show()