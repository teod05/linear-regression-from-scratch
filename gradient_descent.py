import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def gradient_descent(X, y, n_iterations=10000, learning_rate=0.01):
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

    theta = np.array([0.0,0.0]) #m and b
    mse_history = []
    theta_history = []

    for i in range(n_iterations):
        y_pred = np.dot(X, theta)
        
        # Calculate loss
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Compute gradients
        grad = (2/len(X)) * np.dot(X.T, (y_pred - y))

        # Update parameters
        theta = theta - learning_rate * grad
        
        theta_history.append(theta.copy())



    return mse_history, theta, theta_history