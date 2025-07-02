import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100) # I generate 100 random x values, most of them are around 0 (for centering), mean=0 and standard deviation = 1
noise = np.random.randn(100) * 2
y = 2 * x + 3 + noise

def loss_function(y,y_pred):
    y_actual = y
    MSE = np.mean(np.square(y_actual - y_pred))
    return MSE


def gradient_descent(x,y):
    m=0
    b=0

    n= 10000
    lr = 0.01

    for i in range(n):
        y_pred = m*x+b
        # I want to get a SSE and compare it if it is smaller than the previous time
        # if it is smaller I'm going into the right direction
        
        #loss function calculation
        MSE_loss = loss_function(y, y_pred)
        print(MSE_loss)

        #computing the gradients
        m_derrivative = (2/n)*np.sum((y_pred - y)*x)
        b_derrivative = 2 * np.mean(y_pred - y)

        #upadting the derrivatie

        m = m - lr * m_derrivative
        b = b - lr * b_derrivative

        
    y_pred = m * x + b
    return y_pred

    print("FINAL" , y_pred)



gradient_descent(x,y)
y_pred = gradient_descent(x,y)

plt.plot(x, 2 * x + 3, color='red', label='True line') #kind of to see what it is without noise
plt.plot(x, y_pred, color='blue', label='Line of (estimated) best fit') #the estimated line of best
plt.scatter(x, y, label="Data Points") 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()