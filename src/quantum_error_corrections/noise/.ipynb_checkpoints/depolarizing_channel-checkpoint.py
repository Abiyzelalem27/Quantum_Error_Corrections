

import numpy as np

def depolarizing_kraus(p):
    """
    Depolarizing channel.

    Meaning:
        With probability p: the qubit becomes more mixed (random Pauli noise)
        With probability 1-p: stays unchanged

    Kraus operators:
        E0 = sqrt(1 - 3p/4) * I
        E1 = sqrt(p/4) * X
        E2 = sqrt(p/4) * Y
        E3 = sqrt(p/4) * Z
    """

    # Identity and Pauli operators
    I = np.eye(2, dtype=complex)

    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)

    Y = np.array([[0, -1j],
                  [1j, 0]], dtype=complex)

    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - 3*p/4) * I
    E1 = np.sqrt(p/4) * X
    E2 = np.sqrt(p/4) * Y
    E3 = np.sqrt(p/4) * Z

    return [E0, E1, E2, E3]
