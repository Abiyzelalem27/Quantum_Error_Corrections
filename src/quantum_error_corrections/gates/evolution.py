
import numpy as np


def evolve(state, U):
    """
    Evolves a quantum state using unitary U.
    Works for state vector or density matrix.
    """
    if state.ndim == 1:
        return U @ state
    elif state.ndim == 2:
        return U @ state @ U.conj().T
    else:
        raise ValueError("State must be a vector or density matrix.")
