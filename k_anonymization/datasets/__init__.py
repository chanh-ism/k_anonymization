# +
import json
from functools import cached_property
from os import path as os_path

from pandas import read_csv
from .type import Dataset
# -



ADULT = Dataset("adult")
MINI_CRIME = Dataset("mini_crime")
MINI_PATIENT = Dataset("mini_patient")

__all__ = ["Dataset"] + [x.name.upper() for x in Dataset.all_datasets]
