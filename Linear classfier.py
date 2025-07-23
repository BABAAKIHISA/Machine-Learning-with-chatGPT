import numpy as np
import matplotlib.pyplot as plt

# input data and label
y = np.array([
    [1, 7],
    [2, 5],
    [3, 6],
    [5, 6.5],
    [6, 3],
    [-1, -6],
    [-2, -3],
    [-3, -7],
    [-4, -6.5],
    [-5, -1],
])
y_class = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

alpha = 0.01
max_iteration = 1000
a = np.zeros(2)  # Weight vector
a_0 = 0.0        # bias

for iteration in range(max_iteration):
   for number in range(len(y)):
        j = a @ y[number] + a_0 - y_class[number]
        a -= alpha * j * y[number]
        a_0 -= alpha * j


print("Weight vector a:", a)
print("bias a_0:", a_0)

# visualization
plt.figure(figsize=(6, 6))
plt.scatter(y[y_class == -1, 0], y[y_class == -1, 1], color='blue', label='Class -1')
plt.scatter(y[y_class == 1, 0], y[y_class == 1, 1], color='red', label='Class +1')

y1 = np.linspace(-5, 10, 100)
y2 = -(a[0] * y1 + a_0) / a[1]
plt.plot(y1, y2, color='black', label='Decision boundary')

plt.legend()
plt.show()
