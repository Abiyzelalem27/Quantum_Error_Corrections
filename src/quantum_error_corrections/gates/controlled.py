

from .basic_gates import I
from .projectors import P0, P1
from .multi_qubit import U_N_qubits


def controlled_gate(U, control, target, N):
    """
    Controlled-U gate on an N-qubit register:
    C_U = P0(control) ⊗ I + P1(control) ⊗ U(target)
    """
    if control == target:
        raise ValueError("Control and target must be different.")

    P0_ops = [P0 if i == control else I for i in range(N)]
    P1_ops = [P1 if i == control else U if i == target else I for i in range(N)]

    return U_N_qubits(P0_ops) + U_N_qubits(P1_ops)
