"""
Add random noises to the original value to ensure `Pk`-anonymity.

Probabilistic algorithms (a.k.a. `Pk`-anonymization) randomizes the values
of QID attributes such that the probability of identifying a record reduces
to at most 1/`k`.
"""

from .perturbation import Perturbation

__all__ = ["Perturbation"]
