import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import matplotlib

from matplotlib.animation import FFMpegWriter

matplotlib.use('TkAgg')

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi  # coupling strength
zeta = 0.1 * 2 * np.pi  # coupling strength of ion-laser field
epsilon = 0.1 * 2 * np.pi  # cavity-laser field coupling strength
w0 = 1.0 * 2 * np.pi  # laser frequency

# Hilbert space dimension for the cavity
n_cavity = 75

# Operators
sigma_minus = tensor(qeye(n_cavity), sigmam())  # atomic lowering operator
sigma_plus = tensor(qeye(n_cavity), sigmap())  # atomic raising operator
sigma_z = tensor(qeye(n_cavity), sigmaz())  # atomic z operator
a = tensor(destroy(n_cavity), qeye(2))  # cavity annihilation operator

# Times for which the state should be evaluated
times = np.linspace(0, 100, 100)  # time array
tau = wc * times

# Define time-dependent coefficients as strings for QuTiP
H_drive = [
    [sigma_minus, f"{zeta/wc} * exp(1j * {w0/wc} * t)"],
    [sigma_plus, f"{zeta/wc} * exp(-1j * {w0/wc} * t)"],
    [a.dag(), f"{epsilon/wc} * exp(-1j * {w0/wc} * t)"],
    [a, f"{epsilon/wc} * exp(1j * {w0/wc} * t)"]
]

# Hamiltonian with rotating wave approximation and driving fields
H_theoretical = [a.dag() * a + 0.5 * (wa/wc) * sigma_z + (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)] + H_drive

# Initial state: atom in the ground state, cavity in a ground state
psi0 = tensor(basis(n_cavity, 0), basis(2, 0))

# Solve the Schrödinger equation
result = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
output = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[])

n_c = result.expect[0]
n_a = result.expect[1]

# Plotting results
fig, axes = plt.subplots(1, 1, figsize=(10, 6))
axes.plot(times, n_c, label="Cavity photon number")
axes.plot(times, n_a, label="Atom excitation number")
axes.grid(True)
axes.legend(loc='best')
axes.set_xlabel("Time (arbitrary units)")
axes.set_ylabel("Occupation number")
axes.set_title("Occupation number with driving field")
plt.savefig('driven_JC_model.png')
plt.show()

# Animation of the Wigner function (if necessary)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25.0)
xvec = np.linspace(-3, 3, 200)

writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

rho_cavity = [ptrace(rho, 0) for rho in output.states]
fig, ani = anim_wigner(rho_cavity, xvec, xvec, projection='3d', colorbar=True, fig=fig, ax=ax)
ani.save('wigner_function_animation_driven.mp4', writer='ffmpeg', fps=15)

plt.show()
