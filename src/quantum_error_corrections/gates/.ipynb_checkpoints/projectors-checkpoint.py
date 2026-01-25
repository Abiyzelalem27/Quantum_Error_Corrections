
import numpy as np
from .basic_gates import I

P0 = np.array([[1, 0],
               [0, 0]], dtype=complex)

P1 = np.array([[0, 0],
               [0, 1]], dtype=complex)


def projectors(dim: int):
    """
    Generate computational basis projectors {|i><i|}
    """
    proj_list = []
    for i in range(dim):
        ket = np.zeros(dim, dtype=complex)
        ket[i] = 1
        P = np.outer(ket, ket.conj())
        proj_list.append(P)
    return proj_list
