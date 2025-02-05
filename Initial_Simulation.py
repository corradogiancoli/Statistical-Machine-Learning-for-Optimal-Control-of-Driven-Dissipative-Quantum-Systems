# Sample of simulation using Qutip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import sigmam, sigmap, sigmaz
from qutip import (basis, destroy, mesolve, qeye,tensor)
from scipy.signal import find_peaks

matplotlib.use('TkAgg')

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency 
g = 0.05 * 2 * np.pi # coupling strength


# Hilbert space dimension for the cavity
n_cavity = 75

# Operators
sigma_minus = tensor(qeye(n_cavity), sigmam()) # atomic lowering operator
sigma_plus = tensor(qeye(n_cavity), sigmap())   # atomic raising operator
sigma_z = tensor(qeye(n_cavity), sigmaz())       # atomic z operator
a = tensor(destroy(n_cavity), qeye(2))   # cavity annihilation operator


# Times for which the state should be evaluated
times = np.linspace(0, 25, 50000)

tau = wc * times


# Hamiltonian with rotating wave approximation
H_theoretical =  a.dag() * a + 0.5 * (wa/wc) * sigma_z +  (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)

H_experimental = wc * H_theoretical


# Initial state: superposition of state
psi0 = (tensor(basis(n_cavity, 2), basis(2, 0)) + tensor(basis(n_cavity, 1), basis(2, 1)))/(np.sqrt(2))

# Solve the Schrodinger equation
result = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])

n_c = result.expect[0]
n_a = result.expect[1]

#psi_array = psi0.full()
fig, axes = plt.subplots(1, 1, figsize=(10, 6))
axes.plot(tau, n_c, label="Cavity")
axes.plot(tau, n_a, label="Atom Excitation")
axes.grid(True)
axes.legend(loc='best')
axes.set_xlabel("Time (arb. units)")
axes.set_ylabel("Photon number")
axes.set_title("Initial States: (1,g) + (0, e)")

plt.savefig('JC_model.png')
plt.show()

# Fourier Transform
fft_result = np.fft.fft(n_c)
fft_freq = np.fft.fftfreq(n_c.size, d=(tau[1] - tau[0]) / wc)

# Plotting frequency-domain response
fig, ax = plt.subplots()
ax.plot(fft_freq, np.abs(fft_result))
#ax.set_xlim([0, wc])  # Limiting to positive frequencies and within the cavity frequency range
ax.set_xlabel('Frequency (arb. units)')
ax.set_ylabel('Amplitude')

# Find the peak in the FFT
positive_freq_indices = np.where(fft_freq > 0.1)  # Consider only positive frequencies
peak_freq_index = np.argmax(np.abs(fft_result[positive_freq_indices]))  # Index of the max in the positive freq domain
peak_freq = fft_freq[positive_freq_indices][peak_freq_index]  # The frequency corresponding to the peak
peak_amplitude = np.abs(fft_result[positive_freq_indices][peak_freq_index])  # The amplitude at the peak

ax.scatter([peak_freq], [peak_amplitude], color='red')  # Mark the peak frequency
ax.set_xlim([0, 0.5*wc])  # Limiting to positive frequencies and within the cavity frequency range
ax.set_xlabel('Frequency (rad/s)')
ax.set_ylabel('Amplitude')
ax.set_title('Fourier Transform of Cavity Photon Number')
plt.show()

print(f"Peak Frequency: {peak_freq} rad/s, Amplitude: {peak_amplitude}")


#Iterate and verify model

n_list = np.arange(1,11,1)

theo_freq = g * np.sqrt(n_list) #theoretical values

freq_list = []


for n in n_list:
    psi = tensor(basis(n_cavity, n), basis(2, 0)) #+ tensor(basis(n_cavity, n-1), basis(2, 1))
    psi = psi.unit()
    output = mesolve(H_theoretical, psi, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
    n_c = output.expect[0]

    peaks, _ = find_peaks(n_c, height=0)  # You might need to adjust the 'height' parameter based on your data

    # Calculate time points of peaks
    peak_times = tau[peaks]
    periods = np.diff(peak_times)
    average_period = np.mean(periods)
    frequency = 1 / average_period if average_period != 0 else 0

    freq_list.append(frequency * np.pi * wc)

print(freq_list)
plt.figure()
plt.scatter(n_list, freq_list, marker = 'x', label = 'experimental frequency')
plt.scatter(n_list, theo_freq, color = 'red', marker = 'x', label = 'theoretical prediction')
plt.grid(True)
plt.legend(loc='best')
plt.show()