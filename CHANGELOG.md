

# Changelog

All notable changes to this project will be documented in this file.

---

## [2026-01-28]
### Added
- Completed Quantum Error Correction simulation notebooks.
- Implemented 3-qubit **bit-flip code**:
  - Encoding verification
  - Syndrome extraction using stabilizers \(Z_1Z_2\), \(Z_2Z_3\)
  - Correction and fidelity experiments
- Implemented 3-qubit **phase-flip code**:
  - Encoding in the X-basis \(|+++\rangle, |---\rangle\)
  - Syndrome extraction using \(X_1X_2\), \(X_2X_3\)
  - Correction and fidelity evaluation
- Added **depolarizing channel discretization** simulation and Monte Carlo verification:
  \[
  (1-p)\rho + p\frac{I}{2}
  =
  \left(1-\frac{3p}{4}\right)\rho
  +\frac{p}{4}(X\rho X + Y\rho Y + Z\rho Z)
  \]
- Added notebooks for advanced QEC topics:
  - Transversal logical gates
  - Knill–Laflamme error correction conditions
  - Stabilizer syndrome extraction demonstrations
  - \([[4,2,2]]\) detecting code
- Added **amplitude damping** and **phase damping** channel notebooks with fidelity plots.

### Updated
- Improved project structure and notebook naming for GitHub organization.
