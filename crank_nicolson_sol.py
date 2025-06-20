import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
r_max = 1.0
dr = 0.02
r = np.arange(0, r_max + dr, dr)
N = len(r)

T = 48.0  # hours
dt = 0.1  # hours
times = np.arange(0, T + dt, dt)
Nt = len(times)

# Physical constants
D = 1e-6 * 3600
k = 1e-4 * 3600
u1_init, u2_init = 0.5, 1.0
u1e, u2e = 1.0, 0.0

def build_CN_matrices(D, k, u_e):
    L = np.zeros((N, N))
    c = np.zeros(N)
    
    for j in range(1, N-1):
        L[j,j-1] = D/dr**2 - D/(2*r[j]*dr)
        L[j,j]   = -2*D/dr**2
        L[j,j+1] = D/dr**2 + D/(2*r[j]*dr)
    
    L[0,0] = -2*D/dr**2
    L[0,1] =  2*D/dr**2
    

    L[N-1, N-2] = 2*D/dr**2
    L[N-1, N-1] = -2*D/dr**2 - 2*k/dr
    c[N-1] = 2*k/dr * u_e
    
    I = np.eye(N)
    M1 = I - 0.5*dt*L
    M2 = I + 0.5*dt*L
    return M1, M2, c

M1_1, M2_1, c1 = build_CN_matrices(D, k, u1e)
M1_2, M2_2, c2 = build_CN_matrices(D, k, u2e)

# Time-stepping
u1 = np.full(N, u1_init)
u2 = np.full(N, u2_init)
U1 = np.zeros((Nt, N))
U2 = np.zeros((Nt, N))

for n in range(Nt):
    U1[n] = u1
    U2[n] = u2
    rhs1 = M2_1 @ u1 + dt * c1
    rhs2 = M2_2 @ u2 + dt * c2
    u1 = np.linalg.solve(M1_1, rhs1)
    u2 = np.linalg.solve(M1_2, rhs2)

plot_times = [12, 24, 36, 48]
indices = [int(t/dt) for t in plot_times]

for idx, t_val in zip(indices, plot_times):
    plt.figure()
    plt.plot(r, U1[idx], label='Water (u1)')
    plt.plot(r, U2[idx], label='Drug (u2)')
    plt.xlabel('r')
    plt.ylabel('Concentration u')
    plt.title(f'Concentration vs r at t = {t_val} hr')
    plt.legend()
    plt.show()

R, T_grid = np.meshgrid(r, times)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T_grid, U1, rstride=10, cstride=10)
ax.set_xlabel('r')
ax.set_ylabel('t (hr)')
ax.set_zlabel('u1 (water)')
ax.set_title('Water Concentration over r and t')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T_grid, U2, rstride=10, cstride=10)
ax.set_xlabel('r')
ax.set_ylabel('t (hr)')
ax.set_zlabel('u2 (drug)')
ax.set_title('Drug Concentration over r and t')
plt.show()
