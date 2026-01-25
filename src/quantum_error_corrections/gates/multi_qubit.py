
import numpy as np
from .basic_gates import I


def U_N_qubits(ops):
    """Tensor product operator from list of single-qubit operators."""
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U


def U_one_gate(V, i, N):
    """Apply a single-qubit gate V on qubit i in N-qubit register."""
    ops = [I] * N
    ops[i] = V
    return U_N_qubits(ops)


def U_two_gates(V, W, i, j, N):
    """Apply two single-qubit gates V (on i) and W (on j)."""
    ops = [I] * N

    if i == j:
        ops[i] = V @ W
    else:
        ops[i] = V
        ops[j] = W

    return U_N_qubits(ops)
