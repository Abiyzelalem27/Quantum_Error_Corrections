

import numpy as np
def amplitude_damping_kraus(gamma):
    """
    Amplitude damping channel (energy loss / relaxation).

    Physical meaning:
        |1> decays to |0> with probability gamma
        Models spontaneous emission / T1 relaxation

    Kraus operators:
        E0 = [[1, 0],
              [0, sqrt(1-gamma)]]

        E1 = [[0, sqrt(gamma)],
              [0, 0]]
    """

    E0 = np.array([[1, 0],
                   [0, np.sqrt(1 - gamma)]], dtype=complex)

    E1 = np.array([[0, np.sqrt(gamma)],
                   [0, 0]], dtype=complex)

    return [E0, E1]
