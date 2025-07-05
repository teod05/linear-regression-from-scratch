import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def momentum(x, y, n_iterations=10000, learning_rate=0.01, beta=0.9):
    """
    Momentum-based gradient descent implementation for linear regression
    
    Args:
        x: input features
        y: target values
        n_iterations: number of training iterations
        learning_rate: learning rate for parameter updates
        beta: momentum coefficient (typically 0.9)
    
    Returns:
        mse_history: list of MSE values at each iteration
        final_m: final slope parameter
        final_b: final intercept parameter
    """
    m = 0
    b = 0
    vm = 0  # velocity for m
    vb = 0  # velocity for b
    mse_history = []

    for i in range(n_iterations):
        y_pred = m * x + b
        
        # Calculate loss
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Compute gradients
        m_derivative = (2/len(x)) * np.sum((y_pred - y) * x)
        b_derivative = 2 * np.mean(y_pred - y)

        # Update velocities (momentum)
        vm = beta * vm + (1 - beta) * m_derivative
        vb = beta * vb + (1 - beta) * b_derivative

        # Update parameters using velocities
        m = m - learning_rate * vm
        b = b - learning_rate * vb

    return mse_history, m, b

