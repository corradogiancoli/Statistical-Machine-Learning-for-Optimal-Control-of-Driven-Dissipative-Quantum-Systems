# Sample of simulation using Qutip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qutip import *
from qutip import (about, basis, destroy, mesolve, ptrace, qeye,
                   tensor, wigner, anim_wigner)
from matplotlib.animation import FFMpegWriter

matplotlib.use('TkAgg')
#plt.ion() 

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi # coupling strength

# Hilbert space dimension for the cavity
n_cavity = 15

# Operators
sigma_minus = tensor(qeye(n_cavity), sigmam()) # atomic lowering operator
sigma_plus = tensor(qeye(n_cavity), sigmap())   # atomic raising operator
a = tensor(destroy(n_cavity), qeye(2))   # cavity annihilation operator

# Hamiltonian with rotating wave approximation
H_theoretical =  a.dag() * a + (wa/wc) * sigma_plus * sigma_minus +  (g/wc) * (a.dag() * sigma_minus + a * sigma_plus)
H_experimental = wc * H_theoretical


# Initial state: atom in the ground state, cavity in a ground state
psi0 = tensor(basis(n_cavity, 0), basis(2, 0))


# Times for which the state should be evaluated
times = np.linspace(0.0, 100, 100)

tau = wc * times

# Solve the Schrodinger equation
result = mesolve(H_theoretical, psi0, tau, c_ops=[], e_ops=[a.dag() * a, sigma_plus * sigma_minus])
output = mesolve(H_theoretical, psi0, tau, [], [])

n_c = result.expect[0]
n_a = result.expect[1]

fig, axes = plt.subplots(1, 1, figsize=(10, 6))
axes.plot(tau, n_c, label="Cavity")
axes.plot(tau, n_a, label="Atom")
axes.grid(True)
axes.legend(loc='best')
axes.set_xlabel("Time (arb. units)")
axes.set_ylabel("Occupation probability")
axes.set_title("Vacuum Rabi oscillations")

plt.savefig('vacuum_rabi_oscillations.png')
plt.show()

writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=1.0)
xvec = np.linspace(-3, 3, 200)

rho_cavity = [ptrace(rho, 0) for rho in output.states]

fig, ani = anim_wigner(rho_cavity, xvec, xvec, projection='3d',
                       colorbar=True, fig=fig, ax=ax)


ani.save('wigner_function_animation.mp4', writer=writer)

plt.show()





