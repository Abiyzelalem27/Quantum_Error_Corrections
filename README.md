# Quantum Error Corrections (QEC) using Python + NumPy

This repository contains a **step-by-step educational implementation** of core ideas in  
**Quantum Error Correction (QEC)** using **Python **.

Quantum computers are very sensitive to noise (decoherence, imperfect gates).  
QEC provides methods to **detect and correct errors** while preserving the quantum information.

This repository is mainly prepared for **learning + internship portfolio** purposes.

---

## Files

### Notebooks (`notebooks/`)
The notebooks explain and demonstrate QEC concepts in a simple way:

- `01_classical_to_quantum.ipynb`  
  Introduction: why error correction is needed in quantum computing.

- `02_two_qubit_detection_code.ipynb`  
  Basic idea of error detection using redundancy.

- `03_three_qubit_bitflip_code.ipynb`  
  Implementation + demo of the **3-qubit bit-flip code**.

- `04_three_qubit_phaseflip_code.ipynb`  
  Implementation + demo of the **3-qubit phase-flip code**.

- `05_shor_code_9qubit.ipynb`  
  Implementation + demo of the **Shor code (9 qubits)**.

---

### Source Code (`src/quantum_error_corrections/`)

#### `states/`
Quantum state preparation:
- basis states
- Bell states

#### `noise/`
Noise channels used to simulate realistic errors:
- bit-flip channel
- phase-flip channel
- depolarizing channel
- amplitude damping
- phase damping

#### `qec/`
Quantum error correction codes and logic:
- syndrome extraction
- bit-flip 3-qubit code
- phase-flip 3-qubit code
- Shor 9-qubit code
- correction rules

#### `utils/`
Helper functions:
- tensor products
- fidelity calculation

