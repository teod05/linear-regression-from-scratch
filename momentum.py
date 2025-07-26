import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def momentum(X, y, n_iterations=10000, learning_rate=0.01, beta=0.9):
    """
    Momentum-based gradient descent implementation for linear regression
    
    Args:
        X: design matrix (shape: (n, 2))
        y: target values
        n_iterations: number of training iterations
        learning_rate: learning rate for parameter updates
        beta: momentum coefficient (typically 0.9)
    
    Returns:
        mse_history: list of MSE values at each iteration
        m: final slope parameter
        b: final intercept parameter
    """
    theta = np.array([0.0,0.0]) #m and b
    velocity_theta = np.array([0.0, 0.0]) #velocity for m and velocity for b
    mse_history = []
    theta_history = []

    for i in range(n_iterations):
        y_pred = np.dot(X, theta)
        
        # Calculates loss
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Computes gradients
        grad = (2/len(X)) * np.dot(X.T, (y_pred - y))

        # Updates the momentum
        velocity_theta = beta * velocity_theta + learning_rate * grad

        # Update parameters using velocities
        theta = theta - velocity_theta

        theta_history.append(theta.copy())

    return mse_history, theta, theta_history

