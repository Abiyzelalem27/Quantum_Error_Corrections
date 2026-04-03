

import numpy as np 
import pytest
from numpy.testing import assert_allclose

from .quantum_error_corrections import ( 
    I, X, Y, Z, H, P0, P1, I8, X1, X2, X3, Z1, Z2, Z3, 
    bit_flip_kraus_nqubits, U_one_gate, U_two_gates, controlled_gate, 
    rotation_gate, U_N_qubits, initial_state, normalize_state, rho, dm, 
    dm_sparse, random_pure_state, doMeasurement, measurement_density_matrix, 
    sample_from_probs, bit_flip_kraus, phase_flip_kraus, depolarizing_kraus, 
    amplitude_damping_kraus, single_qubit_channel_n_register, apply_channel, 
    apply_kraus, apply_kraus_sparse, encode_3_qubit_bit_flip_code, 
    encode_3_qubit_phase_flip_code, syndrome_measurement, 
    syndrome_measurement_bit_flip, syndrome_measurement_phase_flip, 
    correct_bit_flip, correct_phase_flip, recovery_bit_flip, 
    recovery_phase_flip, deutsch_jozsa, E1_rho, deutsch_jozsa_error1, 
    deutsch_jozsa_error2, deutsch_jozsa_error3, deutsch_jozsa_error4, 
    evolve, apply_hadamards, sample_probs, depolarizing_kraus_nqubits, 
    oracle_function, f_constant_0, f_constant_1, f_balanced_parity, 
    measure_probs_first_n, sample_measurements_input, projectors, 
    born_rule_probs, measure_pure_state, rotation_channel, 
    phase_damping_kraus, pauli_kraus_channel, bit_flip_channel_3qubits, 
    bloch_visualization, ket0, ket1, ket_plus, bloch_vector, ket_minus, 
    ket0_sparse, buildSparseGateSingle, bit_flip_kraus_nqubits_sparse, 
    buildSparseCNOT 
) 


def test_projector_operators():
    """
    Verify projector operators:
    - Idempotent: P^2 = P
    - Hermitian: P† = P
    """
    assert np.allclose(P0 @ P0, P0)
    assert np.allclose(P1 @ P1, P1)
    assert np.allclose(P0.conj().T, P0)
    assert np.allclose(P1.conj().T, P1)


def test_I_gate():
    """Identity gate must be unitary"""
    assert np.allclose(I.conj().T @ I, np.eye(2))


def test_X_gate():
    """Pauli-X gate must be unitary and self-inverse"""
    assert np.allclose(X.conj().T @ X, I)
    assert np.allclose(X @ X, I)


def test_Y_gate():
    """Pauli-Y gate must be unitary and self-inverse"""
    assert np.allclose(Y.conj().T @ Y, I)
    assert np.allclose(Y @ Y, I)


def test_Z_gate():
    """Pauli-Z gate must be unitary and self-inverse"""
    assert np.allclose(Z.conj().T @ Z, I)
    assert np.allclose(Z @ Z, I)


def test_H_gate():
    """Hadamard gate must be unitary and self-inverse"""
    assert np.allclose(H.conj().T @ H, I)
    assert np.allclose(H @ H, I)


def test_U_two_gates():
    """
    Verify embedding of two gates on N qubits:
    - different qubits
    - same qubit
    - reversed indices
    """
    N = 3

    i, j = 0, 2
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)
    assert np.allclose(U_two, expected)

    i = j = 1
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H @ X, i, N)
    assert np.allclose(U_two, expected)

    i, j = 2, 0
    U_two = U_two_gates(H, X, i, j, N)
    expected = U_one_gate(H, i, N) @ U_one_gate(X, j, N)
    assert np.allclose(U_two, expected)


def test_rotation_gate_unitary():
    """Rotation gate must be unitary"""
    theta = np.pi/3
    n = [1,0,0]
    R = rotation_gate(theta, n)
    identity = R @ R.conj().T
    assert np.allclose(identity, I)


def test_U_N_qubits_dimension():
    """U_N_qubits must return correct dimension"""
    U = U_N_qubits([H, H, H])
    assert U.shape == (8,8)


def test_U_one_gate():
    """Single gate embedding dimension"""
    U = U_one_gate(X, 0, 3)
    assert U.shape == (8,8)


def test_rho_trace_one():
    """Density matrix must have trace = 1"""
    psi = ket_plus()
    density = rho([psi], [1])
    assert np.isclose(np.trace(density), 1)


def test_dm_pure_state():
    """Density matrix of |0> state"""
    psi = ket0()
    density = dm(psi)

    expected = np.array([[1,0],[0,0]])
    assert np.allclose(density, expected)


def test_evolve_pure_state():
    """Unitary evolution of pure state"""
    psi = ket0()
    new = evolve(psi, X)
    assert np.allclose(new, ket1())


def test_evolve_density_matrix():
    """Unitary evolution of density matrix"""
    psi = ket0()
    rho0 = dm(psi)
    rho1 = evolve(rho0, X)
    assert np.allclose(rho1, dm(ket1()))


def test_projectors():
    """Projector list properties"""
    P = projectors(2)
    assert len(P) == 2
    assert np.allclose(P[0] @ P[0], P[0])


def test_born_rule():
    """Born rule probability test"""
    psi = ket0()
    rho0 = dm(psi)
    P = projectors(2)
    probs = born_rule_probs(rho0, P)
    assert np.allclose(probs, [1,0])


def test_normalize_state():
    """State normalization"""
    psi = np.array([2,0])
    normalized = normalize_state(psi)
    assert np.isclose(np.linalg.norm(normalized),1)


def test_bit_flip_kraus():
    """Bit-flip Kraus operators"""
    kraus = bit_flip_kraus(0.2)
    assert len(kraus) == 2
    assert kraus[0].shape == (2,2)


def test_apply_channel_trace_preserved():
    """Quantum channel must preserve trace"""
    psi = ket_plus()
    rho0 = dm(psi)
    kraus = bit_flip_kraus(0.3)
    rho1 = apply_channel(rho0, kraus)
    assert np.isclose(np.trace(rho1),1)


def test_bloch_vector():
    """Bloch vector of |0> state"""
    rho0 = dm(ket0())
    bloch = bloch_vector(rho0)
    assert np.allclose(bloch, [0,0,1])


def test_random_state_normalized():
    """Random pure state must be normalized"""
    psi = random_pure_state()
    assert np.isclose(np.linalg.norm(psi),1)


def test_kraus_completeness():
    """Kraus operators completeness relation"""
    kraus = bit_flip_kraus(0.2)
    summation = sum(K.conj().T @ K for K in kraus)
    assert np.allclose(summation, I) 