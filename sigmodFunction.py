import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 50)
y = 1 / (1 + np.exp(-z))
plt.plot(z, y)
plt.show()