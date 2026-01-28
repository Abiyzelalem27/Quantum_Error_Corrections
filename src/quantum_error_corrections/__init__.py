

"""
Quantum Error Corrections package.

Provides quantum operators (gates) and noise channels
implemented using Kraus operators.
"""

# -----------------------
# Operators / gates
# -----------------------
from .operator import (
    I, X, Y, Z, H, S, T, CNOT,
    P0, P1,
    rotation_gate,
    U_N_qubits, U_one_gate, U_two_gates,
    rho, evolve, controlled_gate, projectors, rhoToBlochVec
)

# -----------------------
# Noise channels (Kraus)
# -----------------------
from .channel import (
    apply_kraus,
    bit_flip_kraus,
    phase_flip_kraus,
    depolarizing_kraus,
    amplitude_damping_kraus,
    phase_damping_kraus,
)

__all__ = [
    # operators
    "rhoToBlochVec", "I", "X", "Y", "Z", "H", "S", "T", "CNOT",
    "P0", "P1",
    "rotation_gate",
    "U_N_qubits", "U_one_gate", "U_two_gates",
    "rho", "evolve", "controlled_gate", "projectors",

    # channels
    "apply_kraus",
    "bit_flip_kraus",
    "phase_flip_kraus",
    "depolarizing_kraus",
    "amplitude_damping_kraus",
    "phase_damping_kraus",
]
