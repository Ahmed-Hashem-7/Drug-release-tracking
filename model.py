# =============================================================================
# SBE2250/SBEG108 Course Project: Drug Distribution Modeling
#
# Implementation of a data-driven surrogate model for solving the
# 2D cylindrical diffusion PDE for drug release from a polymer matrix.
#
# Modified to print u1 and u2 radial profiles at a specified time and axial position.
# Outputs native Python floats for easy use in error calculations.
# =============================================================================

import numpy as np
from scipy.integrate import odeint
import time

# --- Main Configuration ---
R0 = 1.0       # Cylinder radius (cm)
ZL = 2.0       # Cylinder length (cm)
U10, U20 = 0.5, 1.0  # Initial concentrations (water, drug)
U1E, U2E = 1.0, 0.0  # External concentrations
DU1, DU2 = 1.0e-6, 1.0e-6  # Diffusivities
KU1, KU2 = 1.0e-1, 1.0e-1  # Mass transfer coeffs

NR, NZ = 11, 11      # Grid points (radial, axial)
T_FINAL_HR = 48      # Total simulation time in hours
T_STEPS = 25         # Number of time points

# Profile selection indices
TIME_INDEX = -1      # Choose last time step by default
Z_INDEX = NZ // 2    # Choose midpoint axial index by default


def generate_and_print_profile():
    print("--- Generating PDE Solution and Extracting Profiles ---")
    start_time = time.time()

    # Grid setup
    dr = R0 / (NR - 1)
    dz = (ZL / 2) / (NZ - 1)
    r_positions = [float(r) for r in np.linspace(0, R0, NR)]
    t_points = np.linspace(0, T_FINAL_HR * 3600, T_STEPS)

    def pde_system_ode(u_flat, t):
        u1 = u_flat[:NR*NZ].reshape((NZ, NR))
        u2 = u_flat[NR*NZ:].reshape((NZ, NR))
        u1t = np.zeros_like(u1)
        u2t = np.zeros_like(u2)

        for i in range(NZ):
            for j in range(NR):
                # Radial
                if j == 0:
                    d2u1_dr2 = 2*(u1[i,j+1] - u1[i,j])/dr**2
                    d2u2_dr2 = 2*(u2[i,j+1] - u2[i,j])/dr**2
                    du1_dr = d2u1_dr2
                    du2_dr = d2u2_dr2
                elif j == NR-1:
                    u1f = u1[i,j-1] + 2*dr*(KU1/DU1)*(U1E - u1[i,j])
                    u2f = u2[i,j-1] + 2*dr*(KU2/DU2)*(U2E - u2[i,j])
                    d2u1_dr2 = (u1f - 2*u1[i,j] + u1[i,j-1])/dr**2
                    d2u2_dr2 = (u2f - 2*u2[i,j] + u2[i,j-1])/dr**2
                    du1_dr = (1/r_positions[j])*(KU1/DU1)*(U1E - u1[i,j])
                    du2_dr = (1/r_positions[j])*(KU2/DU2)*(U2E - u2[i,j])
                else:
                    d2u1_dr2 = (u1[i,j+1] - 2*u1[i,j] + u1[i,j-1])/dr**2
                    d2u2_dr2 = (u2[i,j+1] - 2*u2[i,j] + u2[i,j-1])/dr**2
                    du1_dr = (1/r_positions[j])*(u1[i,j+1] - u1[i,j-1])/(2*dr)
                    du2_dr = (1/r_positions[j])*(u2[i,j+1] - u2[i,j-1])/(2*dr)

                # Axial
                if i == 0:
                    d2u1_dz2 = 2*(u1[i+1,j] - u1[i,j])/dz**2
                    d2u2_dz2 = 2*(u2[i+1,j] - u2[i,j])/dz**2
                elif i == NZ-1:
                    u1f_z = u1[i-1,j] + 2*dz*(KU1/DU1)*(U1E - u1[i,j])
                    u2f_z = u2[i-1,j] + 2*dz*(KU2/DU2)*(U2E - u2[i,j])
                    d2u1_dz2 = (u1f_z - 2*u1[i,j] + u1[i-1,j])/dz**2
                    d2u2_dz2 = (u2f_z - 2*u2[i,j] + u2[i-1,j])/dz**2
                else:
                    d2u1_dz2 = (u1[i+1,j] - 2*u1[i,j] + u1[i-1,j])/dz**2
                    d2u2_dz2 = (u2[i+1,j] - 2*u2[i,j] + u2[i-1,j])/dz**2

                # Time derivatives
                u1t[i,j] = DU1*(d2u1_dr2 + du1_dr + d2u1_dz2)
                u2t[i,j] = DU2*(d2u2_dr2 + du2_dr + d2u2_dz2)

        return np.concatenate([u1t.flatten(), u2t.flatten()])

    # Initial conditions
    u0 = np.concatenate([
        np.full(NZ*NR, U10),
        np.full(NZ*NR, U20)
    ])

    # Solve PDE
    sol = odeint(pde_system_ode, u0, t_points)
    elapsed = time.time() - start_time
    print(f"Solver completed in {elapsed:.2f} s")

    # Reshape solution
    U1 = sol[:, :NR*NZ].reshape((T_STEPS, NZ, NR))
    U2 = sol[:, NR*NZ:].reshape((T_STEPS, NZ, NR))

    # Extract radial profiles
    u1_profile = U1[TIME_INDEX, Z_INDEX, :]
    u2_profile = U2[TIME_INDEX, Z_INDEX, :]

    # Convert to native Python floats
    u1_list = [float(round(val, 3)) for val in u1_profile]
    u2_list = [float(round(val, 3)) for val in u2_profile]

    # Print results
    print(f"Radial positions (r): {r_positions}")
    print(f"u1 profile at time index {TIME_INDEX}, axial index {Z_INDEX}:")
    print(u1_list)
    print(f"u2 profile at time index {TIME_INDEX}, axial index {Z_INDEX}:")
    print(u2_list)

    return u1_list, u2_list


if __name__ == '__main__':
    generate_and_print_profile()
