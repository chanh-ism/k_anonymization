# +
from abc import ABC, abstractmethod

from ..datasets import Dataset


# -

class Algorithm(ABC):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
    ):
        self.k = k
        self.dataset = dataset
        self.anon_data = None

    @property
    def org_data(self):
        return self.dataset.df

    def __reset(self, anonymize_func):
        def wrapper(*args, **kwargs):
            self.anon_data = self.org_data[:]
            self.suppressed_qids = None
            return anonymize_func(self, *args, **kwargs)

        return wrapper

    def __getattribute__(self, name):
        if name == "anonymize":
            func = getattr(type(self), "anonymize")
            return self.__reset(func)
        return object.__getattribute__(self, name)

    @abstractmethod
    def anonymize(self):
        pass


