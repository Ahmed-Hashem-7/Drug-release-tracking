# =============================================================================
# SBE2250/SBEG108 Course Project: Drug Distribution Modeling
#
# Implementation of a data-driven surrogate model for solving the
# 2D cylindrical diffusion PDE for drug release from a polymer matrix.
#
# Method:
# 1. Numerical Solution: Solves the PDE system using the Method of Lines
#    with finite differences and scipy's odeint ODE solver.
# 2. Surrogate Modeling: Trains a feed-forward neural network on the
#    generated numerical data to create a fast predictive model.
#
# Based on: W. E. Schiesser, "Partial Differential Equation Analysis
# in Biomedical Engineering: Case Studies with MATLAB", 2013, Ch. 7.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- Main Configuration ---
# Physical and Geometric Parameters
R0 = 1.0       # Cylinder radius (cm)
ZL = 2.0       # Cylinder length (cm)
U10, U20 = 0.5, 1.0  # Initial concentrations (water, drug)
U1E, U2E = 1.0, 0.0  # External concentrations (water, drug)
DU1, DU2 = 1.0e-6, 1.0e-6 # Diffusivities (cm^2/s)
KU1, KU2 = 1.0e-1, 1.0e-1 # Mass transfer coefficients (cm/s)

# Numerical Solver Parameters
NR, NZ = 11, 11      # Number of grid points (radial, axial)
T_FINAL_HR = 48      # Total simulation time in hours
T_STEPS = 25         # Number of time points to solve for

# Neural Network Parameters
EPOCHS = 500
BATCH_SIZE = 64

# --- PART 1: NUMERICAL SOLVER & DATA GENERATION ---

def generate_pde_solution_data():
    """
    Solves the 2D cylindrical PDE system using the Method of Lines.
    Returns a pandas DataFrame with the solution data.
    """
    print("--- Starting Part 1: Numerical PDE Solution ---")
    start_time = time.time()

    # Grid setup
    dr = R0 / (NR - 1)
    dz = (ZL / 2) / (NZ - 1)
    r_grid = np.linspace(0, R0, NR)
    z_grid = np.linspace(ZL / 2, ZL, NZ)
    t_points = np.linspace(0, T_FINAL_HR * 3600, T_STEPS)

    def pde_system_ode(u_flat, t):
        # The ODE system function for scipy's odeint solver
        u1 = u_flat[:NR * NZ].reshape((NZ, NR))
        u2 = u_flat[NR * NZ:].reshape((NZ, NR))
        u1t, u2t = np.zeros_like(u1), np.zeros_like(u2)

        for i in range(NZ):
            for j in range(NR):
                # Finite Difference Approximations for spatial derivatives
                # Radial derivatives
                if j == 0: # r=0
                    d2u1_dr2 = 2 * (u1[i, j + 1] - u1[i, j]) / dr**2
                    d2u2_dr2 = 2 * (u2[i, j + 1] - u2[i, j]) / dr**2
                    du1_dr_term = d2u1_dr2
                    du2_dr_term = d2u2_dr2
                elif j == NR - 1: # r=R0
                    u1f = u1[i, j - 1] + 2 * dr * (KU1 / DU1) * (U1E - u1[i, j])
                    u2f = u2[i, j - 1] + 2 * dr * (KU2 / DU2) * (U2E - u2[i, j])
                    d2u1_dr2 = (u1f - 2 * u1[i, j] + u1[i, j - 1]) / dr**2
                    d2u2_dr2 = (u2f - 2 * u2[i, j] + u2[i, j - 1]) / dr**2
                    du1_dr_term = (1/r_grid[j]) * (KU1 / DU1) * (U1E - u1[i, j])
                    du2_dr_term = (1/r_grid[j]) * (KU2 / DU2) * (U2E - u2[i, j])
                else: # Interior
                    d2u1_dr2 = (u1[i, j + 1] - 2 * u1[i, j] + u1[i, j - 1]) / dr**2
                    d2u2_dr2 = (u2[i, j + 1] - 2 * u2[i, j] + u2[i, j - 1]) / dr**2
                    du1_dr_term = (1/r_grid[j]) * (u1[i, j + 1] - u1[i, j - 1]) / (2 * dr)
                    du2_dr_term = (1/r_grid[j]) * (u2[i, j + 1] - u2[i, j - 1]) / (2 * dr)

                # Axial derivatives
                if i == 0: # z=ZL/2
                    d2u1_dz2 = 2 * (u1[i + 1, j] - u1[i, j]) / dz**2
                    d2u2_dz2 = 2 * (u2[i + 1, j] - u2[i, j]) / dz**2
                elif i == NZ - 1: # z=ZL
                    u1f = u1[i - 1, j] + 2 * dz * (KU1 / DU1) * (U1E - u1[i, j])
                    u2f = u2[i - 1, j] + 2 * dz * (KU2 / DU2) * (U2E - u2[i, j])
                    d2u1_dz2 = (u1f - 2 * u1[i, j] + u1[i - 1, j]) / dz**2
                    d2u2_dz2 = (u2f - 2 * u2[i, j] + u2[i - 1, j]) / dz**2
                else: # Interior
                    d2u1_dz2 = (u1[i + 1, j] - 2 * u1[i, j] + u1[i - 1, j]) / dz**2
                    d2u2_dz2 = (u2[i + 1, j] - 2 * u2[i, j] + u2[i - 1, j]) / dz**2

                # Full PDE
                u1t[i, j] = DU1 * (d2u1_dr2 + du1_dr_term + d2u1_dz2)
                u2t[i, j] = DU2 * (d2u2_dr2 + du2_dr_term + d2u2_dz2)

        return np.concatenate([u1t.flatten(), u2t.flatten()])

    # Initial conditions
    u1_initial = np.full((NZ, NR), U10)
    u2_initial = np.full((NZ, NR), U20)
    u_initial_flat = np.concatenate([u1_initial.flatten(), u2_initial.flatten()])

    # Run solver
    print("  Solving PDE system numerically...")
    solution = odeint(pde_system_ode, u_initial_flat, t_points)
    print(f"  Numerical solution complete. Time elapsed: {time.time() - start_time:.2f} s")

    # Reshape and format data into a DataFrame
    T, Z, R = np.meshgrid(t_points, z_grid, r_grid, indexing='ij')
    U1 = solution[:, :NR*NZ].reshape(T_STEPS, NZ, NR)
    U2 = solution[:, NR*NZ:].reshape(T_STEPS, NZ, NR)

    df = pd.DataFrame({
        't': T.flatten(), 'r': R.flatten(), 'z': Z.flatten(),
        'u1': U1.flatten(), 'u2': U2.flatten()
    })
    return df

# --- PART 2: SURROGATE MODEL TRAINING ---

def train_surrogate_model(df):
    """
    Trains a surrogate neural network on the PDE solution data.
    """
    print("\n--- Starting Part 2: Surrogate Model Training ---")
    start_time = time.time()

    # Prepare data for training
    features = ['t', 'r', 'z']
    targets = ['u1', 'u2']
    X, y = df[features].values, df[targets].values

    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    X_scaled, y_scaled = x_scaler.fit_transform(X), y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Build the neural network
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    print(f"  Training model for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.1, verbose=1
    )
    print(f"  Training complete. Time elapsed: {time.time() - start_time:.2f} s")

    # Evaluate performance
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Surrogate Model Performance: Test MSE = {test_loss:.4e}")
    return model, x_scaler, y_scaler

# --- PART 3: RESULTS & VISUALIZATION ---

def plot_results(model, x_scaler, y_scaler):
    """
    Generates 2D and 3D plots of the surrogate model's solution.
    """
    print("\n--- Starting Part 3: Generating Visualizations ---")

    # --- 2D Composite Plot (like Figure 7.6) ---
    fig2d, axes2d = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    axes2d = axes2d.flatten()
    plot_times_hr = [12.0, 24.0, 36.0, 48.0]
    plot_z = ZL / 2.0
    r_vals = np.linspace(0, R0, 101)

    for i, t_hr in enumerate(plot_times_hr):
        ax = axes2d[i]
        t_sec = t_hr * 3600

        X_pred_raw = np.array([[t_sec, r, plot_z] for r in r_vals])
        X_pred_scaled = x_scaler.transform(X_pred_raw)
        y_pred_scaled = model.predict(X_pred_scaled, verbose=0)
        y_pred_raw = y_scaler.inverse_transform(y_pred_scaled)

        ax.plot(r_vals, y_pred_raw[:, 0], 'o-', label='u1 (Water)', markersize=4)
        ax.plot(r_vals, y_pred_raw[:, 1], 'x-', label='u2 (Drug)', markersize=4)
        ax.set_title(f'Concentration Profile at t = {t_hr:.0f} hr', fontsize=14)
        ax.set_xlabel('Radius r (cm)', fontsize=12)
        ax.set_ylabel('Concentration', fontsize=12)
        ax.grid(True, linestyle='--')
        ax.legend()
        ax.set_ylim(-0.1, 1.1)

    fig2d.suptitle('Surrogate Model: 2D Radial Profiles over Time', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- 3D Surface Plots (like Figures 7.3 and 7.4) ---
    print("  Generating 3D surface plots...")
    t_plot_3d = np.linspace(0, T_FINAL_HR * 3600, T_STEPS)
    r_plot_3d = np.linspace(0, R0, NR)
    T_grid, R_grid = np.meshgrid(t_plot_3d, r_plot_3d)

    # Prepare input grid for prediction
    Z_grid = np.full_like(T_grid, ZL / 2.0)
    X_pred_3d_raw = np.vstack([T_grid.ravel(), R_grid.ravel(), Z_grid.ravel()]).T
    X_pred_3d_scaled = x_scaler.transform(X_pred_3d_raw)

    # Predict and reshape
    y_pred_3d_scaled = model.predict(X_pred_3d_scaled, verbose=0)
    y_pred_3d_raw = y_scaler.inverse_transform(y_pred_3d_scaled)

    U1_surf = y_pred_3d_raw[:, 0].reshape(R_grid.shape)
    U2_surf = y_pred_3d_raw[:, 1].reshape(R_grid.shape)
    T_grid_days = T_grid / (24 * 3600) # Convert time to days for plotting

    # Plot for Water (u1)
    fig3d_1 = plt.figure(figsize=(12, 8))
    ax1 = fig3d_1.add_subplot(111, projection='3d')
    ax1.plot_surface(R_grid, T_grid_days, U1_surf, cmap='viridis')
    ax1.set_title('3D Surface Plot: Water Concentration (u1)', fontsize=16)
    ax1.set_xlabel('Radius r (cm)', fontsize=12)
    ax1.set_ylabel('Time t (days)', fontsize=12)
    ax1.set_zlabel('Concentration u1', fontsize=12)
    ax1.view_init(elev=20, azim=-135) # Set view angle
    plt.show()

    # Plot for Drug (u2)
    fig3d_2 = plt.figure(figsize=(12, 8))
    ax2 = fig3d_2.add_subplot(111, projection='3d')
    ax2.plot_surface(R_grid, T_grid_days, U2_surf, cmap='plasma')
    ax2.set_title('3D Surface Plot: Drug Concentration (u2)', fontsize=16)
    ax2.set_xlabel('Radius r (cm)', fontsize=12)
    ax2.set_ylabel('Time t (days)', fontsize=12)
    ax2.set_zlabel('Concentration u2', fontsize=12)
    ax2.view_init(elev=20, azim=45) # Set different view angle
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Step 1: Generate training data
    pde_data = generate_pde_solution_data()

    # Step 2: Train the surrogate model
    model, x_scaler, y_scaler = train_surrogate_model(pde_data)

    # Step 3: Save the model
    model.save('drug_surrogate_model.keras')
    print("\n✅ Final model saved to 'drug_surrogate_model.keras'")

    # Step 4: Visualize all results
    plot_results(model, x_scaler, y_scaler)
