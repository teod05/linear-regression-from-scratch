import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def adam(X, y, n_iterations=10000, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Adam optimization for linear regression.
    
    Args:
        X: Design matrix (shape: (n, 2))
        y: Target values (shape: (n,))
        n_iterations: Number of iterations
        learning_rate: Step size
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        epsilon: Small constant for numerical stability
    
    Returns:
        mse_history: List of MSE values
        theta: Final parameter vector [b, m]
        theta_history: List of parameter vectors over iterations
    """
    theta = np.array([0.0, 0.0])
    m = np.zeros(2)  # First moment
    v = np.zeros(2)  # Second moment
    mse_history = []
    theta_history = []
    t = 0

    for i in range(n_iterations):
        y_pred = np.dot(X, theta)
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)
        grad = (2 / len(X)) * np.dot(X.T, (y_pred - y))
        
        t += 1  # Increment before correction
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - (learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))

        theta_history.append(theta.copy())

    return mse_history, theta, theta_history