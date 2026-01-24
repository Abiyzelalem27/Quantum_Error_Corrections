

import numpy as np

def I():
    return np.array([[1, 0],
                     [0, 1]], dtype=complex)

def X():
    return np.array([[0, 1],
                     [1, 0]], dtype=complex)

def Y():
    return np.array([[0, -1j],
                     [1j, 0]], dtype=complex)

def Z():
    return np.array([[1, 0],
                     [0, -1]], dtype=complex)
