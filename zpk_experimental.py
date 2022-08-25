from scipy import signal
import numpy as np

sys = signal.ZerosPolesGain(np.array([[1, 2, 5], [5, 6, 7]]), [3, 4, 4], 0)

print(sys)
print(f"{sys.inputs=}")
print(f"{sys.outputs=}")
