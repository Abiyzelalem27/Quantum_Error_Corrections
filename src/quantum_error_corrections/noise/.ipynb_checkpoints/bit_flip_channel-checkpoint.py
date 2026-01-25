

import numpy as np

def bit_flip_kraus(p):
    """
    Bit flip channel (X noise).

    With probability p: apply X (flip |0> <-> |1>)
    With probability 1-p: do nothing (I)

    Kraus operators:
        E0 = sqrt(1-p) * I
        E1 = sqrt(p)   * X
    """

    # Identity operator
    I = np.eye(2, dtype=complex)

    # Pauli-X operator (bit flip)
    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)

    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X

    return [E0, E1]
