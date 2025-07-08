import numpy as np

def loss_function(y, y_pred):
    """Calculate Mean Squared Error loss"""
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE

def adam(x,y, n_iterations=1000, learning_rate=0.1, beta=0.9, beta_2=0.999):
    """
    xcxxcxcxcxcxcxcxcxcxcxc
    """
    m = 0
    b = 0
    vm = 0
    vb = 0
    mse_history=[]
    v2m = 0
    v2b = 0
    t= 0
    epsilon = 1e-8 

    for i in range(n_iterations):
    
        y_pred = m*x+b

        #calucates MSE
        MSE_loss = loss_function(y, y_pred)
        mse_history.append(MSE_loss)

        # Computes gradients
        m_derivative = (2/len(x)) * np.sum((y_pred - y) * x)
        b_derivative = 2 * np.mean(y_pred - y)

        #updates the momentum
        vm = beta * vm + (1 - beta) * m_derivative
        vb = beta * vb + (1 - beta) * b_derivative

        #updates second momentum
        v2m = beta_2 * v2m + (1- beta_2) * m_derivative**2
        v2b = beta_2 * v2b + (1- beta_2) * b_derivative**2
        #bias corrected first moments
        t+=1
        vm_hat = vm / (1-beta**t)
        vb_hat = vb / (1-beta**t)

        #biast corrected second moments
        v2m_hat = v2m / (1-beta_2**t)
        v2b_hat = v2b / (1-beta_2**t)

        #updating the values

        m = m - learning_rate * (vm_hat / (np.sqrt(v2m_hat) + epsilon))
        b = b - learning_rate * (vb_hat / (np.sqrt(v2b_hat) + epsilon))

    return mse_history, m, b


        


