

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
# Noise channels + tools
# -----------------------
from .channel import (
    # Kraus application
    apply_kraus,

    # Basic Kraus channels
    bit_flip_kraus,
    phase_flip_kraus,
    depolarizing_kraus,
    amplitude_damping_kraus,
    phase_damping_kraus,
    

    # ✅ Added important missing functions
    ket0, ket1,
    dm,
    random_pure_state,

    fidelity,
    fidelity_pure_rho,
    sample_min_fidelity,

    pauli_kraus_channel,
    normalize,
    random_qubit_state,
    encode,
    majority_vote_correct,
    decode_to_single_qubit,
    monte_carlo_fidelity,
    pauli_string,
    apply_bitflips,
    
    

    # Q3 channels (bit-flip code fidelity)
    E1_rho,
    E3_effective_rho,
    min_fidelity_bitflip_channel,
    min_fidelity_three_qubit_code,
    improvement_condition,

)

__all__ = [
    # -----------------------
    # Operators
    # -----------------------
    "rhoToBlochVec",
    "I", "X", "Y", "Z", "H", "S", "T", "CNOT",
    "P0", "P1",
    "rotation_gate",
    "U_N_qubits", "U_one_gate", "U_two_gates",
    "rho", "evolve", "controlled_gate", "projectors",

    # -----------------------
    # Channels
    # -----------------------
    "apply_kraus",
    "fidelity",
    "apply_bitflips",
    "pauli_string",
    "monte_carlo_fidelity",
    "normalize",
    "random_qubit_state",
    "encode",
    "majority_vote_correct",
    "decode_to_single_qubit",
    "bit_flip_kraus",
    "phase_flip_kraus",
    "depolarizing_kraus",
    "amplitude_damping_kraus",
    "phase_damping_kraus",
    "improvement_condition",


    # -----------------------
    # ✅ Added utilities / states
    # -----------------------
    "ket0", "ket1",
    "dm",
    "random_pure_state",

    "fidelity_pure_rho",
    "sample_min_fidelity",

    "pauli_kraus_channel",

    # -----------------------
    # ✅ Q3 helper channels
    # -----------------------
    "E1_rho",
    "E3_effective_rho",
    "min_fidelity_bitflip_channel",
    "min_fidelity_three_qubit_code",
]
