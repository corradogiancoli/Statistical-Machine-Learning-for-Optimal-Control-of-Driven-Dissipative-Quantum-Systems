import numpy as np
import matplotlib.pyplot as plt
from qutip import (mesolve, qeye, tensor, fock, Qobj, fidelity)
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence
from skopt.utils import use_named_args


# System parameters
wc = 1.0 * 2 * np.pi  # cavity frequency
wa = 1.0 * 2 * np.pi  # atom frequency
g = 0.05 * 2 * np.pi # cavity-atom coupling
f = 0.5 * 2 * np.pi # laser-atom coupling
w0 = 0.8 * 2 * np.pi # laser driving frequency
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

delta_c = wc - w0
delta_a = wa - w0

times = np.arange(0, 10, 1e-5)
tau = wc * times

H_driven_RWA = delta_c * a.dag() * a + 0.5 * delta_a * sigma_z + g * (sigma_plus * a + sigma_minus * a.dag())
+ 0.5 * f * (sigma_plus + sigma_minus)

H_driven_RWA = H_driven_RWA/wc

psi = tensor(fock(n_cavity, 3), spin_up)

result = mesolve(H_driven_RWA, psi, tau, c_ops = [], e_ops = [a.dag() * a, sigma_plus * sigma_minus])

n_c = result.expect[0]
n_a = result.expect[1]

plt.figure()
plt.plot(tau, n_c, label = 'Cavity')
plt.plot(tau, n_a, label = 'Atom Excited State')
plt.grid(True)
plt.xlabel('Time (arb. units)')
plt.ylabel('Fock state')
plt.legend(loc='best')
plt.show()

psi_den = psi * psi.dag()

output = mesolve(H_driven_RWA, psi_den, tau, c_ops = [], e_ops = [])

target_state = tensor(fock(n_cavity, 1), spin_down)

target_den = target_state * target_state.dag()

fid = fidelity(output.states[-1], target_den)
print("Fidelity:", fid)

# Search space for Bayesian Optimization
space  = [Real(0.1 * 2 * np.pi, 1.0 * 2 * np.pi, name='f'),
          Real(0.5 * 2 * np.pi, 1.5 * 2 * np.pi, name='w0')]

# General objective function that accepts named arguments
def objective_general(f, w0):
    delta_c = wc - w0
    delta_a = wa - w0
    H = (delta_c * a.dag() * a + 0.5 * delta_a * sigma_z + 
         g * (sigma_plus * a + sigma_minus * a.dag()) +
         0.5 * f * (sigma_plus + sigma_minus))
    H /= wc
    result = mesolve(H, psi_den, tau, c_ops=[], e_ops=[])
    final_rho = result.states[-1]
    fid = fidelity(target_den, final_rho)
    return -fid  # return negative fidelity for minimization

def objective(x):
    return objective_general(x[0], x[1])

# Perform Bayesian Optimization
res = gp_minimize(objective, space, n_calls=30, random_state=0, verbose=True)

# Now generate data for heatmap using the general function
f_range = np.linspace(0.1, 1, 50) * 2 * np.pi
w0_range = np.linspace(0.5, 1.5, 50) * 2 * np.pi
fidelity_map = np.zeros((50, 50))

for i, f_val in enumerate(f_range):
    for j, w0_val in enumerate(w0_range):
        fidelity_map[i, j] = -objective_general(f=f_val, w0=w0_val)  # negative to convert back to positive fidelity

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(fidelity_map, extent=(0.5, 1.5, 0.1, 1), origin='lower', aspect='auto', cmap='viridis')
plt.colorbar(label='Fidelity')
plt.title('Fidelity Heatmap')
plt.xlabel('$w_0 / 2\pi$')
plt.ylabel('$f / 2\pi$')
plt.show()

# Best parameters found
print(f"Optimal f: {res.x[0]/(2*np.pi)} Hz, Optimal w0: {res.x[1]/(2*np.pi)} Hz")