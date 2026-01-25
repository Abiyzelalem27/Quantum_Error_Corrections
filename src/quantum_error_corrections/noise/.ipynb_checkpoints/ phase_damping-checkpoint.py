

import numpy as np

def phase_damping_kraus(lmbda):

    """
    Phase damping channel (dephasing / loss of coherence).

    Physical meaning:
        No energy loss, but off-diagonal terms decrease.
        Models decoherence / T2 effects.

    Kraus operators:
        E0 = [[1, 0],
              [0, sqrt(1-lambda)]]

        E1 = [[0, 0],
              [0, sqrt(lambda)]]
    """

    E0 = np.array([[1, 0],
                   [0, np.sqrt(1 - lmbda)]], dtype=complex)

    E1 = np.array([[0, 0],
                   [0, np.sqrt(lmbda)]], dtype=complex)

    return [E0, E1]
    
