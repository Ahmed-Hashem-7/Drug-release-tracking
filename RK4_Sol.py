import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# True values
true_u1 = np.array([0.761, 0.764, 0.774, 0.790, 0.811, 0.838, 0.868, 0.900, 0.934, 0.968, 1.000])
true_u2 = np.array([0.479, 0.472, 0.453, 0.421, 0.378, 0.325, 0.265, 0.199, 0.131, 0.064, 0.000])

# Parameters
r_max = 1.0
dr = 0.02
r = np.arange(0, r_max + dr, dr)
N = len(r)
T = 48.0
dt = 0.01  #delta h
times = np.arange(0, T + dt, dt)
Nt = len(times)

D = 1e-6 * 3600
k = 1e-4 * 3600
u1_init, u2_init = 0.5, 1.0
u1e, u2e = 1.0, 0.0

# RHS definition for diffusion with radial symmetry and Robin BC
def RHS(u, D, k, u_e):
    dudt = np.zeros_like(u)
    for j in range(1, N - 1):
        dudt[j] = D * ((u[j - 1] - 2 * u[j] + u[j + 1]) / dr ** 2 +
                       (u[j + 1] - u[j - 1]) / (2 * r[j] * dr))
    dudt[0] = D * (2 * (u[1] - u[0]) / dr ** 2)
    dudt[-1] = D * (2 * (u[-2] - u[-1]) / dr ** 2 - (2 * k / (D * dr)) * (u[-1] - u_e))
    return dudt

# RK4 implementation
def RK4_step(u, dt, D, k, u_e):
    k1 = RHS(u, D, k, u_e)
    k2 = RHS(u + 0.5 * dt * k1, D, k, u_e)
    k3 = RHS(u + 0.5 * dt * k2, D, k, u_e)
    k4 = RHS(u + dt * k3, D, k, u_e)
    return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

# Time-stepping
u1 = np.full(N, u1_init)
u2 = np.full(N, u2_init)
U1 = np.zeros((Nt, N))
U2 = np.zeros((Nt, N))

for n in range(Nt):
    U1[n] = u1
    U2[n] = u2
    u1 = RK4_step(u1, dt, D, k, u1e)
    u2 = RK4_step(u2, dt, D, k, u2e)

# 2D Plots
plot_times = [12, 24, 36, 48]
indices = [int(t / dt) for t in plot_times]

for idx, t_val in zip(indices, plot_times):
    plt.figure()
    plt.plot(r, U1[idx], label='Water (u1)')
    plt.plot(r, U2[idx], label='Drug (u2)')
    plt.xlabel('r (cm)')
    plt.ylabel('Concentration')
    plt.title(f'Concentration profiles at t = {t_val} hr')
    plt.legend()
    plt.grid()
    plt.show()

# 3D Surface Plots
R, T_grid = np.meshgrid(r, times)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T_grid, U1, cmap='viridis')
ax.set_xlabel('r (cm)')
ax.set_ylabel('t (hr)')
ax.set_zlabel('u1 (Water)')
ax.set_title('Water Concentration')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T_grid, U2, cmap='plasma')
ax.set_xlabel('r (cm)')
ax.set_ylabel('t (hr)')
ax.set_zlabel('u2 (Drug)')
ax.set_title('Drug Concentration')
plt.show()

# Error computation
def compute_errors(sim_u, true_u):
    abs_error = np.abs(sim_u - true_u)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.zeros_like(abs_error)
        mask = true_u != 0
        rel_error[mask] = abs_error[mask] / np.abs(true_u[mask])
    mse = np.mean(abs_error**2)
    rmse = np.sqrt(mse)
    return {
        'Absolute Error': abs_error,
        'Relative Error': rel_error,
        'MSE': mse,
        'RMSE': rmse
    }

sample_indices = [int(i * (N - 1) / 10) for i in range(11)]
final_u1 = U1[-1, sample_indices]
final_u2 = U2[-1, sample_indices]

# Error results
u1_errors = compute_errors(final_u1, true_u1)
u2_errors = compute_errors(final_u2, true_u2)

print("Errors for u1:")
for key, val in u1_errors.items():
    print(f"{key}: {val}")

print("\nErrors for u2:")
for key, val in u2_errors.items():
    print(f"{key}: {val}")
