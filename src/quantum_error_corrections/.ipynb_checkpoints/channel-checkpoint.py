



import numpy as np

def apply_kraus(rho, kraus_ops):
    """
    Apply a quantum noise channel using Kraus operators.

    E(rho) = Σ_k (E_k rho E_k†)

    Parameters:
        rho (2x2 numpy array): input density matrix
        kraus_ops (list): list of Kraus matrices

    Returns:
        (2x2 numpy array): output density matrix
    """

    # Start with a zero matrix (same shape as rho)
    out = np.zeros_like(rho, dtype=complex)

    # Sum over all Kraus operators
    for E in kraus_ops:
        out += E @ rho @ E.conj().T   # E * rho * E†

    return out


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
