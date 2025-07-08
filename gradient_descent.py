import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def gradient_descent(x, y, n_iterations=1000, learning_rate=0.1):
    """
    Basic gradient descent implementation for linear regression
    
    Args:
        x: input features
        y: target values
        n_iterations: number of training iterations
        learning_rate: learning rate for parameter updates
    
    Returns:
        mse_history: list of MSE values at each iteration
        final_m: final slope parameter
        final_b: final intercept parameter
    """
    m = 0
    b = 0
    mse_history = []

    for i in range(n_iterations):
        y_pred = m * x + b
        
        # Calculate loss
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Compute gradients
        m_derivative = (2/len(x)) * np.sum((y_pred - y) * x)
        b_derivative = 2 * np.mean(y_pred - y)

        # Update parameters
        m = m - learning_rate * m_derivative
        b = b - learning_rate * b_derivative

    return mse_history, m, b 