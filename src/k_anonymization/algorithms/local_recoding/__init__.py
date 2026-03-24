"""
Split records into groups and apply anonymization locally in each group.

Local recoding algorithms first attempts to divide the data into smaller
groups (e.g., partitions or clusters) of at least `k` closely-related records.
Then, anonymization is applied to records within each group, typically by
assigning an identical representative value for each QID attribute.
This results in several anonymized clusters of size ≥ `k` whose records are
identical (with respect to QID); the anonymized data is simply a
concatenation of these anonymized groups, thus guaranteeing `k`-anonymity.
"""

from .kmember import KMember
from .local_recoding_algorithm import (
    GroupAnonymization,
    GroupAnonymizationBuiltIn,
    LocalRecodingAlgorithm,
)
from .mondrian import ClassicMondrian
from .oka.oka import OKA

__all__ = [
    "LocalRecodingAlgorithm",
    "GroupAnonymization",
    "GroupAnonymizationBuiltIn",
    "ClassicMondrian",
    "KMember",
    "OKA",
]
