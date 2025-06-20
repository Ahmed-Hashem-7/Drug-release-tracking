import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

# True values
true_u1 = np.array([0.761, 0.764, 0.774, 0.790, 0.811, 0.838, 0.868, 0.900, 0.934, 0.968, 1.000])
true_u2 = np.array([0.479, 0.472, 0.453, 0.421, 0.378, 0.325, 0.265, 0.199, 0.131, 0.064, 0.000])

# Model parameters
u10, u20 = 0.5, 1.0
u1e, u2e = 1.0, 0.0
r0, zL = 1.0, 2.0
Du1, Du2 = 1e-6, 1e-6
ku1, ku2 = 1e-1, 1e-1

# Grid
nr, nz = 11, 11
dr = r0 / (nr - 1)
dz = zL / (2 * (nz - 1))
r = np.linspace(0, r0, nr)
z = zL / 2 + np.arange(nz) * dz

t_days = np.linspace(0, 2, 5)
t_eval = t_days * 3600 * 24

# Initial condition
u0 = np.concatenate((np.full(nz * nr, u10), np.full(nz * nr, u20)))


def pde_rhs(t, u):
    """MOL right‐hand side for both u1 and u2."""
    u1 = u[:nz * nr].reshape(nz, nr)
    u2 = u[nz * nr:].reshape(nz, nr)
    u1t = np.zeros_like(u1)
    u2t = np.zeros_like(u2)

    for i in range(nz):
        for j in range(nr):
            # radial derivatives
            if j == 0:
                u1r = 2 * (u1[i, 1] - u1[i, 0]) / dr ** 2
                u2r = 2 * (u2[i, 1] - u2[i, 0]) / dr ** 2
                u1rr = u1r
                u2rr = u2r
            elif j == nr - 1:
                u1r = (ku1 / Du1) * (u1e - u1[i, j]) / r[j]
                u2r = (ku2 / Du2) * (u2e - u2[i, j]) / r[j]
                u1f = u1[i, j - 1] + 2 * dr * ku1 / Du1 * (u1e - u1[i, j])
                u2f = u2[i, j - 1] + 2 * dr * ku2 / Du2 * (u2e - u2[i, j])
                u1rr = (u1f - 2 * u1[i, j] + u1[i, j - 1]) / dr ** 2
                u2rr = (u2f - 2 * u2[i, j] + u2[i, j - 1]) / dr ** 2
            else:
                u1r = (u1[i, j + 1] - u1[i, j - 1]) / (2 * dr) / r[j]
                u2r = (u2[i, j + 1] - u2[i, j - 1]) / (2 * dr) / r[j]
                u1rr = (u1[i, j + 1] - 2 * u1[i, j] + u1[i, j - 1]) / dr ** 2
                u2rr = (u2[i, j + 1] - 2 * u2[i, j] + u2[i, j - 1]) / dr ** 2

            # axial derivatives
            if i == 0:
                u1zz = 2 * (u1[i + 1, j] - u1[i, j]) / dz ** 2
                u2zz = 2 * (u2[i + 1, j] - u2[i, j]) / dz ** 2
            elif i == nz - 1:
                u1f_z = u1[i - 1, j] + 2 * dz * ku1 / Du1 * (u1e - u1[i, j])
                u2f_z = u2[i - 1, j] + 2 * dz * ku2 / Du2 * (u2e - u2[i, j])
                u1zz = (u1f_z - 2 * u1[i, j] + u1[i - 1, j]) / dz ** 2
                u2zz = (u2f_z - 2 * u2[i, j] + u2[i - 1, j]) / dz ** 2
            else:
                u1zz = (u1[i + 1, j] - 2 * u1[i, j] + u1[i - 1, j]) / dz ** 2
                u2zz = (u2[i + 1, j] - 2 * u2[i, j] + u2[i - 1, j]) / dz ** 2
            # PDEs
            u1t[i, j] = Du1 * (u1rr + u1r + u1zz)
            u2t[i, j] = Du2 * (u2rr + u2r + u2zz)

    return np.concatenate((u1t.ravel(), u2t.ravel()))


#solutionwith Radau
sol = solve_ivp(pde_rhs, (0, t_eval[-1]), u0,
                method='Radau', t_eval=t_eval, rtol=1e-4, atol=1e-4)

# Extract final radial profiles at z = zL/2
u1_fin = sol.y[:nz * nr, -1].reshape(nz, nr)[0]
u2_fin = sol.y[nz * nr:, -1].reshape(nz, nr)[0]

# percent relative errors
with np.errstate(divide='ignore', invalid='ignore'):
    err_u1 = np.abs(u1_fin - true_u1) / np.abs(true_u1) * 100
    err_u2 = np.abs(u2_fin - true_u2) / np.abs(true_u2) * 100

err_u2[true_u2 == 0] = np.nan

# Print error table
print("r\tError u1 (%)\tError u2 (%)")
for ri, e1, e2 in zip(r, err_u1, err_u2):
    print(f"{ri:.2f}\t{e1:8.4f}\t{'' if np.isnan(e2) else f'{e2:8.4f}'}")

# 2D plot comparison at 48 h
plt.figure()
plt.plot(r, true_u1, 'x-', label='u1 (true)')
plt.plot(r, u1_fin, 'o--', label='u1 (Radau)')
plt.plot(r, true_u2, 'x-', label='u2 (true)')
plt.plot(r, u2_fin, 'o--', label='u2 (Radau)')
plt.xlabel('r (cm)')
plt.ylabel('Concentration')
plt.title('Radial Profiles at t = 48 h')
plt.legend()
plt.show()

# 3D surface plots with t on x-axis and r on y-axis
T_mesh, R_mesh = np.meshgrid(t_days, r)
u1_surf = sol.y[:nz * nr, :].reshape(nz, nr, -1)[0]  # shape (nr, nt)
u2_surf = sol.y[nz * nr:, :].reshape(nz, nr, -1)[0]

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(T_mesh, R_mesh, u1_surf, edgecolor='k', linewidth=0.3)
ax1.set_xlabel('t (days)')
ax1.set_ylabel('r (cm)')
ax1.set_zlabel('u1')
ax1.set_title('u1 over (t, r)')
ax1.view_init(elev=30, azim=-60)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(T_mesh, R_mesh, u2_surf, edgecolor='k', linewidth=0.3)
ax2.set_xlabel('t (days)')
ax2.set_ylabel('r (cm)')
ax2.set_zlabel('u2')
ax2.set_title('u2 over (t, r)')
ax2.view_init(elev=30, azim=-60)

plt.tight_layout()
plt.show()
