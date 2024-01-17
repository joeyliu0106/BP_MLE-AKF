from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 11)
y = np.cos(-x**2/9.0)
y[-1] = y[0]  # make the sequence periodic
x_fine = np.linspace(0, 10, 1000)
bc_types = ['not-a-knot', 'periodic', 'clamped', 'natural']

for s in bc_types:
    f = CubicSpline(x, y, bc_type=s)
    plt.plot(x_fine, f(x_fine), '-', alpha=0.8, label=s)
    print(f(x_fine))


print(x)
print(y)
plt.plot(x, y, 'ko')
plt.grid(linestyle='--', alpha=0.3)
plt.legend()
plt.show()