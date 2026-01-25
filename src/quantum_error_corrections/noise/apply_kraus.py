

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
