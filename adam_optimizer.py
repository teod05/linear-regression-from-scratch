import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def adam(X,y, n_iterations=10000, learning_rate=0.01, beta=0.9, beta_2=0.999):
    """
    
    """
    theta = np.array([0.0,0.0])
    theta_velocity = np.array([0.0,0.0])
    mse_history=[]
    theta_history = []
    theta_2_velocity = np.array([0.0,0.0])
    t= 0
    epsilon = 1e-8 

    for i in range(n_iterations):
    
        y_pred = np.dot(X, theta)

        #calucates MSE
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Computes gradients
        grad = (2/len(X)) * np.dot(X.T, (y_pred - y))

        #updates the momentum
        theta_velocity = beta * theta_velocity + learning_rate * grad

        #updates second momentum
        v2m = beta_2 * theta_2_velocity + (1- beta_2) * grad**2
        #bias corrected first moments
        t+=1
        theta_hat = theta_velocity / (1-beta**t)

        #biast corrected second moments
        theta_2_hat = theta_2_velocity / (1-beta_2**t)
        #updating the values

        theta = theta - learning_rate * (theta_hat / (np.sqrt(theta_2_hat) + epsilon))

        theta_history.append(theta.copy())

    return mse_history, theta, theta_history


        


