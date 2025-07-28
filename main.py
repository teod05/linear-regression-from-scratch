import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import gradient_descent
from momentum import momentum
from adam_optimizer import adam
import time

# Generate centered data
x_centered = np.random.randn(100)  # Mean=0, std=1
x_centered = (x_centered - np.mean(x_centered)) / np.std(x_centered)  # Standardize
X_centered = np.column_stack([np.ones_like(x_centered), x_centered])
noise_centered = np.random.randn(100) * 2
y_centered = 2 * x_centered + 3 + noise_centered

# Generate non-centered data
x_noncentered = np.random.randn(100) * 2 + 10  # Mean=10, std=2
x_noncentered = (x_noncentered - np.mean(x_noncentered)) / np.std(x_noncentered)  # Standardize
X_noncentered = np.column_stack([np.ones_like(x_noncentered), x_noncentered])
noise_noncentered = np.random.randn(100) * 2
y_noncentered = 2 * x_noncentered + 3 + noise_noncentered

# Function to run optimizations and return results
def run_optimization(X, y):
    start_time = time.time()
    mse_gd, theta_gd, theta_gd_history = gradient_descent(X, y)
    gd_time = time.time() - start_time

    start_time = time.time()
    mse_mom, theta_mom, theta_mom_history = momentum(X, y)
    mom_time = time.time() - start_time

    start_time = time.time()
    mse_adam, theta_adam, theta_adam_history = adam(X, y)
    adam_time = time.time() - start_time

    return (mse_gd, theta_gd, theta_gd_history, mse_mom, theta_mom, theta_mom_history,
            mse_adam, theta_adam, theta_adam_history)

# Run for centered data
centered_results = run_optimization(X_centered, y_centered)

# Run for non-centered data
noncentered_results = run_optimization(X_noncentered, y_noncentered)

# Visualization for Centered Data
plt.figure(figsize=(12, 5))

# Plot 1: Data and fitted lines (Centered)
plt.subplot(1, 2, 1)
plt.scatter(x_centered, y_centered, alpha=0.6, label="Data Points")
plt.plot(x_centered, 2 * x_centered + 3, color='red', label='True line')
plt.plot(x_centered, centered_results[1][1] * x_centered + centered_results[1][0], color='blue', label='Gradient Descent')
plt.plot(x_centered, centered_results[4][1] * x_centered + centered_results[4][0], color='green', label='Momentum')
plt.plot(x_centered, centered_results[7][1] * x_centered + centered_results[7][0], color='purple', label='Adam')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Results (Centered)")

# Plot 2: Convergence comparison (Centered)
plt.subplot(1, 2, 2)
plt.plot(centered_results[0][:1000], label='Gradient Descent', alpha=0.7)
plt.plot(centered_results[3][:1000], label='Momentum', alpha=0.7)
plt.plot(centered_results[6][:1000], label='Adam', alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.title("Convergence Comparison (Centered, First 1000 Iterations)")
plt.yscale('log')

plt.tight_layout()
plt.show()

# Visualization for Non-Centered Data
plt.figure(figsize=(12, 5))

# Plot 1: Data and fitted lines (Non-Centered)
plt.subplot(1, 2, 1)
plt.scatter(x_noncentered, y_noncentered, alpha=0.6, label="Data Points")
plt.plot(x_noncentered, 2 * x_noncentered + 3, color='red', label='True line')
plt.plot(x_noncentered, noncentered_results[1][1] * x_noncentered + noncentered_results[1][0], color='blue', label='Gradient Descent')
plt.plot(x_noncentered, noncentered_results[4][1] * x_noncentered + noncentered_results[4][0], color='green', label='Momentum')
plt.plot(x_noncentered, noncentered_results[7][1] * x_noncentered + noncentered_results[7][0], color='purple', label='Adam')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Results (Non-Centered)")

# Plot 2: Convergence comparison (Non-Centered)
plt.subplot(1, 2, 2)
plt.plot(noncentered_results[0][:1000], label='Gradient Descent', alpha=0.7)
plt.plot(noncentered_results[3][:1000], label='Momentum', alpha=0.7)
plt.plot(noncentered_results[6][:1000], label='Adam', alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.title("Convergence Comparison (Non-Centered, First 1000 Iterations)")
plt.yscale('log')

plt.tight_layout()
plt.show()

# Error Surface (Centered Data)
def loss_function(m, b, x, y):
    return np.mean((y - (m * x + b))**2)

m_range = np.linspace(-1, 5, 100)
b_range = np.linspace(0, 6, 100)
m_grid, b_grid = np.meshgrid(m_range, b_range)
loss_grid_centered = np.zeros(m_grid.shape)

for i in range(m_grid.shape[0]):
    for j in range(m_grid.shape[1]):
        loss_grid_centered[i, j] = loss_function(m_grid[i, j], b_grid[i, j], x_centered, y_centered)

plt.figure(figsize=(8, 6))
plt.contour(m_grid, b_grid, loss_grid_centered, levels=20, cmap='viridis')
plt.plot([th[1] for th in centered_results[2][:1000]], [th[0] for th in centered_results[2][:1000]], 'b.-', label='Gradient Descent', alpha=0.7)
plt.plot([th[1] for th in centered_results[5][:1000]], [th[0] for th in centered_results[5][:1000]], 'g.-', label='Momentum', alpha=0.7)
plt.plot([th[1] for th in centered_results[8][:1000]], [th[0] for th in centered_results[8][:1000]], 'm.-', label='Adam', alpha=0.7)
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.title('Error Surface with Parameter Paths (Centered)')
plt.legend()
plt.colorbar(label='Loss')
plt.show()

# Error Surface (Non-Centered Data)
loss_grid_noncentered = np.zeros(m_grid.shape)
for i in range(m_grid.shape[0]):
    for j in range(m_grid.shape[1]):
        loss_grid_noncentered[i, j] = loss_function(m_grid[i, j], b_grid[i, j], x_noncentered, y_noncentered)

plt.figure(figsize=(8, 6))
plt.contour(m_grid, b_grid, loss_grid_noncentered, levels=20, cmap='viridis')
plt.plot([th[1] for th in noncentered_results[2][:1000]], [th[0] for th in noncentered_results[2][:1000]], 'b.-', label='Gradient Descent', alpha=0.7)
plt.plot([th[1] for th in noncentered_results[5][:1000]], [th[0] for th in noncentered_results[5][:1000]], 'g.-', label='Momentum', alpha=0.7)
plt.plot([th[1] for th in noncentered_results[8][:1000]], [th[0] for th in noncentered_results[8][:1000]], 'm.-', label='Adam', alpha=0.7)
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')
plt.title('Error Surface with Parameter Paths (Non-Centered)')
plt.legend()
plt.colorbar(label='Loss')
plt.show()