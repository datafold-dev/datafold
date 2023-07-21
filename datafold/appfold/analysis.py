import matplotlib.pyplot as plt
import numpy as np

K_own = np.load("K_own.npy")
K_other = np.load("koopman_matrix.npy")

plt.matshow(K_own, vmin=-1, vmax=1)
plt.title("K_own")
plt.matshow(K_other, vmin=-1, vmax=1)
plt.title("K_other")


plt.show()
