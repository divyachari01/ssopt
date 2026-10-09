"""ssopt — SUPERSTRUCTURE λ optimisation with an energy surrogate.

Pipeline:  ion → CI configs → ssin → structx (energies only) → match to NIST
           → multi-output GP of E_k(λ) → minimise RMS on the surrogate
           → verify → refit.   A-values are a read-only check at the end.

All energies are in Rydberg unless a function name says otherwise.
"""
__version__ = "0.6.8"
