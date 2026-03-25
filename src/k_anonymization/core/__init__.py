"""
Core (Abstract) classes.
"""

from .algorithm import Algorithm
from .dataset import Dataset, SampleDataset
from .frame import ITableDF
from .hierarchy import HierarchiesDict, Hierarchy
from .parallel import Parallel

__all__ = [
    "Algorithm",
    "Dataset",
    "SampleDataset",
    "ITableDF",
    "HierarchiesDict",
    "Hierarchy",
    "Parallel",
]
