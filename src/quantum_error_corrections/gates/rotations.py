
import numpy as np
from .basic_gates import I, X, Y, Z


def rotation_gate(theta, n):
    """
    General single-qubit rotation gate.

    This function implements a unitary rotation of a single qubit
    by an angle `theta` around an axis `n` on the Bloch sphere.

    The rotation generator is constructed as N = n · σ,
    where σ = (X, Y, Z) are the Pauli matrices.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    n : tuple of floats
        Rotation axis (nx, ny, nz).
    """
    nx, ny, nz = n
    N = nx * X + ny * Y + nz * Z
    R = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * N
    return R