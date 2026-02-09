
import numpy as np
from .operator import I, X, Y, Z

# ============================================================
# APPLY KRAUS MAP
# ============================================================

def apply_kraus(rho, kraus_ops):
    """
    Apply a quantum noise channel using Kraus operators.

    E(rho) = Σ_k (E_k rho E_k†)

    Parameters
    ----------
    rho : numpy.ndarray
        Input density matrix (2x2)

    kraus_ops : list[numpy.ndarray]
        List of Kraus operators {E_k}

    Returns
    -------
    numpy.ndarray
        Output density matrix (2x2)
    """
    out = np.zeros_like(rho, dtype=complex)

    for E in kraus_ops:
        out += E @ rho @ E.conj().T

    return out


# ============================================================
# STANDARD SINGLE-QUBIT NOISE CHANNELS (KRAUS)
# ============================================================

def bit_flip_kraus(p):
    """
    Bit flip channel (X noise).

    Kraus operators:
        E0 = sqrt(1-p) I
        E1 = sqrt(p)   X
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * X
    return [E0, E1]


def phase_flip_kraus(p):
    """
    Phase flip channel (Z noise).

    Kraus operators:
        E0 = sqrt(1-p) I
        E1 = sqrt(p)   Z
    """
    E0 = np.sqrt(1 - p) * I
    E1 = np.sqrt(p) * Z
    return [E0, E1]


def amplitude_damping_kraus(gamma):
    """
    Amplitude damping channel.

    Physical meaning:
        |1> -> |0> with probability gamma

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
    Phase damping channel (dephasing).

    Physical meaning:
        Off-diagonal terms decay but populations stay unchanged.
    """
    E0 = np.array([[1, 0],
                   [0, np.sqrt(1 - lmbda)]], dtype=complex)

    E1 = np.array([[0, 0],
                   [0, np.sqrt(lmbda)]], dtype=complex)

    return [E0, E1]


def depolarizing_kraus(p):
    """
    Depolarizing channel.

    Kraus operators:
        E0 = sqrt(1 - 3p/4) I
        E1 = sqrt(p/4) X
        E2 = sqrt(p/4) Y
        E3 = sqrt(p/4) Z
    """
    E0 = np.sqrt(1 - 3*p/4) * I
    E1 = np.sqrt(p/4) * X
    E2 = np.sqrt(p/4) * Y
    E3 = np.sqrt(p/4) * Z
    return [E0, E1, E2, E3]


# ============================================================
# BASIC STATES + DENSITY MATRIX
# ============================================================

def ket0():
    """
    Return the computational basis state |0⟩.

    |0⟩ = [1, 0]^T
    """
    return np.array([1, 0], dtype=complex)


def ket1():
    """
    Return the computational basis state |1⟩.

    |1⟩ = [0, 1]^T
    """
    return np.array([0, 1], dtype=complex)


def dm(psi):
    """
    Construct a density matrix from a pure state |psi⟩.

    ρ = |psi⟩⟨psi|
    """
    return np.outer(psi, psi.conj())


def random_pure_state(rng=None):
    """
    Generate a random single-qubit pure state |psi⟩.

    Useful for Monte-Carlo simulations.
    """
    rng = np.random.default_rng() if rng is None else rng
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = v / np.linalg.norm(v)
    return v


# ============================================================
# FIDELITY FUNCTIONS (Q3)
# ============================================================

def fidelity_pure_rho(psi, rho):
    """
    Fidelity between pure state |psi⟩ and density matrix rho:

        F(|psi>, rho) = <psi|rho|psi>
    """
    return np.real(np.vdot(psi, rho @ psi))


def sample_min_fidelity(channel_fn, p, n_samples=5000, seed=0):
    """
    Monte-Carlo estimate of the minimum fidelity over random states.
    """
    rng = np.random.default_rng(seed)
    Fmin = 1.0

    for _ in range(n_samples):
        psi = random_pure_state(rng)
        rho_out = channel_fn(psi, p)
        F = fidelity_pure_rho(psi, rho_out)
        Fmin = min(Fmin, F)

    return Fmin


# ============================================================
# GENERAL PAULI CHANNEL (IMPORTANT FOR SHOR / DEPOLARIZING)
# ============================================================

def pauli_kraus_channel(pX, pY, pZ):
    """
    General Pauli channel:

    E(rho) = pI*rho + pX XrhoX + pY YrhoY + pZ ZrhoZ

    where:
        pI = 1 - (pX+pY+pZ)
    """
    pI = 1 - (pX + pY + pZ)
    if pI < 0:
        raise ValueError("Probabilities must satisfy pX+pY+pZ <= 1")

    E0 = np.sqrt(pI) * I
    E1 = np.sqrt(pX) * X
    E2 = np.sqrt(pY) * Y
    E3 = np.sqrt(pZ) * Z

    return [E0, E1, E2, E3]


# ============================================================
# CHANNELS FOR Q3 (E1 and EFFECTIVE E3 LOGICAL CHANNEL)
# ============================================================

def E1_rho(psi, p):
    """
    Q3: One-qubit bit flip channel:
        E1(rho) = (1-p)rho + pXrhoX
    """
    rho = dm(psi)
    return (1 - p) * rho + p * (X @ rho @ X)


def E3_effective_rho(psi, p):
    """
    Q3: Effective logical channel after 3-qubit bit-flip code correction.

    It successfully recovers the original state if 0 or 1 errors occur.

    Success probability:
        q = (1-p)^3 + 3p(1-p)^2
    """
    q = (1 - p)**3 + 3 * p * (1 - p)**2
    rho = dm(psi)
    return q * rho + (1 - q) * (X @ rho @ X)


def min_fidelity_bitflip_channel(p):
    """
    Q3(a) theoretical minimum fidelity:
        F_min = 1 - p
    """
    return 1 - p


def min_fidelity_three_qubit_code(p):
    """
    Q3(b) lower bound on fidelity after correction:
        F_min >= (1-p)^3 + 3p(1-p)^2
    """
    return (1 - p)**3 + 3 * p * (1 - p)**2
    

def improvement_condition():
    """
    Q3(c) Condition for error correction improvement:

    We need:
        Fmin(E3) > Fmin(E1)

    Which gives:
        p < 1/2
    """
    return "Error correction improves fidelity when p < 1/2"

def normalize(v):
    n = np.linalg.norm(v)
    return v / n

def random_qubit_state():
    # random complex 2-vector
    v = (np.random.randn(2) + 1j*np.random.randn(2)).astype(complex)
    return normalize(v)

def encode(psi):  # psi is (2,) normalized
    # alpha|000> + beta|111>
    alpha, beta = psi[0], psi[1]
    out = np.zeros(8, dtype=complex)
    out[0] = alpha         # |000>
    out[7] = beta          # |111>
    return out


def majority_vote_correct(state3):
    # project to closest codeword by majority of bits:
    # basis |000> and |111> only; we "collapse" amplitude groups
    alpha_eff = 0+0j
    beta_eff = 0+0j
    for idx, amp in enumerate(state3):
        bits = [(idx >> 2) & 1, (idx >> 1) & 1, idx & 1]
        ones = sum(bits)
        if ones <= 1:
            alpha_eff += amp   # treat as |000> component after correction
        else:
            beta_eff += amp    # treat as |111> component after correction
    out = np.zeros(8, dtype=complex)
    out[0] = alpha_eff
    out[7] = beta_eff
    return out

def decode_to_single_qubit(state3):
    # since corrected state is alpha|000> + beta|111>, decode to (alpha, beta)
    return np.array([state3[0], state3[7]], dtype=complex)

def fidelity(psi, rho_or_pure):
    # psi is pure state vector (2,)
    # rho_or_pure can be a pure vector (2,) or density (2,2)
    if rho_or_pure.ndim == 1:
        phi = rho_or_pure
        return abs(np.vdot(psi, phi))**2
    else:
        rho = rho_or_pure
        return np.real(np.vdot(psi, rho @ psi))

def monte_carlo_fidelity(p, trials=20000):
    F_vals = []
    for _ in range(trials):
        psi = random_qubit_state()
        enc = encode(psi)
        noisy = apply_bitflips(enc, p)
        corr = majority_vote_correct(noisy)
        dec = decode_to_single_qubit(corr)
        # normalize decoded (because our correction step sums amplitudes)
        dec = normalize(dec)
        F_vals.append(fidelity(psi, dec))
    return float(np.mean(F_vals))

def pauli_string(n, ops):
    """
    n   : total number of qubits
    ops : dict {site: 'X','Y','Z'}
          site is 1-indexed
    """
    pauli_map = {
        'I': I,
        'X': X,
        'Y': Y,
        'Z': Z
    }

    result = 1
    for i in range(1, n + 1):
        if i in ops:
            result = np.kron(result, pauli_map[ops[i]])
        else:
            result = np.kron(result, I)

    return result

def apply_bitflips(state, p):
    noisy = state.copy()
    for i in range(3):  # 3-qubit repetition
        if np.random.rand() < p:
            noisy = apply_X(noisy, i)  # your X-on-qubit-i function
    return noisy
