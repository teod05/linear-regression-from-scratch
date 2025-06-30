import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100) # I generate 100 random x values, most of them are around 0 (for centering), mean=0 and standard deviation = 1
noise = np.random.randn(100) * 2
y = 2 * x + 3 + noise


#poltting data

plt.plot(x, 2 * x + 3, color='red', label='True line') #kind of to see what it is without noise


plt.scatter(x, y, label="Data Points") 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show

