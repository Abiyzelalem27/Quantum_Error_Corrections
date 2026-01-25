
import numpy as np


def rho(states, probabilities):
    """Construct density matrix from pure states and probabilities."""
    return sum(p * np.outer(psi, psi.conj())
               for psi, p in zip(states, probabilities))
