# Sample of simulation using Qutip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import (basis, destroy, mesolve, qeye,tensor, SESolver, sigmaz, sigmam, sigmap, fock)
from scipy.signal import find_peaks

matplotlib.use('TkAgg')

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency 
g = 0.5 * 2 * np.pi # coupling strength

# Hilbert space dimension for the cavity
n_cavity = 75

# Operators
sigma_minus = tensor(qeye(n_cavity), destroy(2)) # atomic lowering operator
sigma_plus = sigma_minus.dag()   # atomic raising operator
sigma_z = tensor(qeye(n_cavity), sigmaz())     # atomic z operator
a = tensor(destroy(n_cavity), qeye(2))   # cavity annihilation operator

# Times for which the state should be evaluated
times = np.linspace(0, 25, 50000)

tau = wc * times


# Hamiltonian with rotating wave approximation
H_theoretical =  a.dag() * a + 0.5 * (wa/wc) * sigma_minus*sigma_plus +  (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)

H_experimental = wc * H_theoretical


# Initial state: superposition of state
psi0 = tensor(fock(n_cavity, 1), fock(2, 1)) + tensor(fock(n_cavity, 2), fock(2, 0))


# Solve the Schrodinger equation
result = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
#result_alt =  mesolve(H_experimental, psi0, times, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
#solver = SESolver(H_theoretical)
#result = solver.run(psi0, tau, e_ops = [a.dag() * a, sigma_plus * sigma_minus])

n_c = result.expect[0]
n_a = result.expect[1]

fig, axes = plt.subplots(1, 1, figsize=(10, 6))#
axes.plot(result.times, n_c, label="Cavity")
axes.plot(result.times, n_a, label="Atom Excitation")
axes.grid(True)
axes.legend(loc='best')
axes.set_xlabel("Time (arb. units)")
axes.set_xlim(0,5)
axes.set_ylabel("Photon number")
axes.set_title("Initial States: (2,g) + (1,e)")
#plt.savefig('JC_model.png')
plt.show()


#Iterate and verify model

n_list = np.arange(1,11,1)

theo_freq = g * np.sqrt(n_list)#theoretical values

freq_list = []



for n in n_list:
    psi = tensor(fock(n_cavity, n), fock(2,0)) + tensor(fock(n_cavity, n-1), fock(2,1))
    output = mesolve(H_theoretical, psi, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
    #output_alt = mesolve(H_experimental, psi, times, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
    n_c = output.expect[0]
    n_a = output.expect[1]
    #n_a_alt = output_alt.expect[1]

    peaks, _ = find_peaks(n_a, height=0.5) 
    #peaks_alt, _ = find_peaks(n_a_alt, height=0.5)
    #fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    #axes.plot(output.times, n_c, label="Cavity Photon Number")
    #axes.plot(output.times, n_a, label="Atom Excitation")
    #axes.scatter(tau[peaks], np.array(n_a)[peaks], color='red', s=10, marker='o', label='Peaks')  # Mark peaks
    #axes.grid(True)
    #axes.legend(loc='best')
    #axes.set_xlabel("Time (arb.units)")
    #axes.set_ylabel("Photon number")
    #axes.set_title(f"Photon and Atom Excitation Dynamics for n={n}")
    #plt.show()


    
    # Calculate time points of peaks
    peak_times = tau[peaks]
    #peak_times_alt = times[peaks_alt]
    periods = np.diff(peak_times)
    #print(f'{n}:',periods)
    #periods_alt = np.diff(peak_times_alt)
    average_period = np.mean(periods)
    #average_period_alt = np.mean(periods_alt) * wc
    #print(f'The theo period is {average_period} and the experimental period is {average_period_alt}')
    error_period = np.std(periods)
    frequency = 1 / average_period if average_period != 0 else 0

    freq_list.append(frequency * 2 * np.pi * wc)


plt.figure()
plt.scatter(n_list, freq_list, marker = 'x', label = 'experimental frequency')
plt.scatter(n_list, theo_freq, color = 'red', marker = 'x', label = 'theoretical prediction')
plt.xlabel('Fock state of cavity')
plt.ylabel('Frequency (arb. units)')
plt.xscale('log')  # Setting log scale for x-axis
plt.yscale('log')
plt.grid(True)
plt.legend(loc='best')
plt.savefig('Rabi_oscillations_2.png')
plt.show()
