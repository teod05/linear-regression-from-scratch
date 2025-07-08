import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import gradient_descent
from momentum import momentum
import time
from adam_optimizer import adam

# Generate synthetic data
x = np.random.randn(100)  # 100 random x values, mean=0, std=1
noise = np.random.randn(100) * 2
y = 2 * x + 3 + noise

# Runs GD

start_time = time.time()
mse_gd, m_gd, b_gd = gradient_descent(x, y)
gd_time = time.time() - start_time

# Runw Momentum
start_time = time.time()
mse_mom, m_mom, b_mom = momentum(x, y)
mom_time = time.time() - start_time

# Runs Adam
start_time = time.time()
mse_adm, m_adm, b_adm = adam(x, y)
adm_time = time.time() - start_time

# Print comparison results
# print("Gradient Descent MSE:")
# for i in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
#     print(f"Iteration {i}: {mse_gd[i]}")

# print("\nMomentum MSE:")
# for i in [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
#     print(f"Iteration {i}: {mse_mom[i]}")

# print("\nAdam MSE:")
# for i in [1, 2, 50, 100, 200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9999]:
#     print(f"Iteration {i}: {mse_adm[i]}")


print(f"Gradient Descent time: {gd_time:.4f} seconds")
print(f"Momentum time: {mom_time:.4f} seconds")

# # Visualization 
plt.figure(figsize=(12, 5))

# Plot 1: Data and fitted lines
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.6, label="Data Points")
plt.plot(x, 2 * x + 3, color='red', label='True line')
plt.plot(x, m_gd * x + b_gd, color='blue', label='Gradient Descent')
plt.plot(x, m_mom * x + b_mom, color='green', label='Momentum')
plt.plot(x, m_adm * x + b_adm, color='purple', label='Adam')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Results")

# Plot 2: Convergence comparison
plt.subplot(1, 2, 2)
plt.plot(mse_gd[:1000], label='Gradient Descent', alpha=0.7)
plt.plot(mse_mom[:1000], label='Momentum', alpha=0.7)
plt.plot(mse_adm[:1000], label='Adam', alpha=0.7)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.title("Convergence Comparison (First 1000 Iterations)")
plt.yscale('log')

plt.tight_layout()
plt.show()