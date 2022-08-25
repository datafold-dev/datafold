import control
import numpy as np

from scipy.signal import StateSpace




A=np.array([[1,2 ], [3,4]])
B=np.array([[1],[1]])

A = np.array([[1, 1], [0, 1]])
B = np.array([[1, 0], [0, 1]])
C = np.array([[1, 0]])
D = np.array([[0, 0]])

control.StateSpace(A, B, C, D)