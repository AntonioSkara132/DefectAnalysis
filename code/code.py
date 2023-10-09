import matplotlib.pyplot as plt
import numpy as np

ax = plt.axes(projection="3d")

x_data = np.arange(0, 50, 0.1)
y_data = np.arange(0, 50, 0.1)

X, Y = np.meshgrid(x_data, y_data)
Z = -1*X - 1*Y

ax.plot_surface(X, Y, Z)

plt.show()























