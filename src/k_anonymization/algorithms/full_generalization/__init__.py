"""
Replace domains of QID attributes based on their given generalizations.

Full-domain generalization algorithms replaces the entire domain of values
of QID attributes with their higher-level form(s) of abstraction, based on
the `generalization hierarchies`. This results in many original values
become identical as they share a common ancestor on their generalization
hierarchy, which helps achieving `k`-anonymity.
"""

from .datafly import Datafly

__all__ = ["Datafly"]
