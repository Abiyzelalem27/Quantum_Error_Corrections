

import numpy as np

def phase_flip_kraus(p):
    """
    Phase flip channel (Z noise).

    With probability p: apply Z (flip phase of |1>)
    With probability 1-p: do nothing (I)

    Kraus operators:
        E0 = sqrt(1-p) * I
        E1 = sqrt(p)   * Z
    """

    # Identity operator
    I = np.eye(2, dtype=complex)

    # Pauli-Z operator (phase flip)
    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z

    return [E0, E1]
