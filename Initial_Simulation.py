# Sample of simulation using Qutip

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip import (about, basis, destroy, mesolve, ptrace, qeye,
                   tensor, wigner, anim_wigner)
from matplotlib.animation import FFMpegWriter

# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 # coupling strength

# Hilbert space dimension for the cavity
n_cavity = 15

# Operators
sigma_minus = tensor(qeye(n_cavity), sigmam()) # atomic lowering operator
sigma_plus = tensor(qeye(n_cavity), sigmap())   # atomic raising operator
a = tensor(destroy(n_cavity), qeye(2))   # cavity annihilation operator

# Hamiltonian with rotating wave approximation
H = wc * a.dag() * a + wa * sigma_plus * sigma_minus +  g * (a.dag() * sigma_minus + a * sigma_plus)
#print('Shaoe of H:', H.shape)

# Initial state: atom in the ground state, cavity in a ground state
psi0 = tensor(basis(n_cavity, 0), basis(2, 0))
#print('Shape of psi0:', psi0.shape)

# Times for which the state should be evaluated
times = np.linspace(0.0, 100, 100)

# Solve the Schrodinger equation
result = mesolve(H, psi0, times, [a.dag() * a, sigma_plus * sigma_minus])

writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# Save the animation

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=0)
xvec = np.linspace(-3, 3, 200)

rho_cavity = [ptrace(rho, 0) for rho in result.states]

fig, ani = anim_wigner(rho_cavity, xvec, xvec, projection='3d',
                       colorbar=True, fig=fig, ax=ax)


ani.save('wigner_function_animation.mp4', writer=writer)

# close an auto-generated plot and animation
plt.close()

