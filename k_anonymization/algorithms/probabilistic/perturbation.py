# +
import numpy as np

from k_anonymization.algorithms.type import Algorithm
from k_anonymization.datasets import Dataset


# -

class Perturbation(Algorithm):
    def __init__(
        self,
        dataset: Dataset,
        k: int,
        seed: int = None,
    ):
        super().__init__(dataset, k)

        self.seed = seed
        self.cat_qids = []
        self.cat_uniques = {}
        self.num_qids = []
        self.num_ranges = {}
        for idx, qid in enumerate(self.dataset.qids):
            if self.dataset.is_categorical[idx] is True:
                self.cat_qids.append(qid)
                self.cat_uniques[qid] = self.org_data[qid].unique()
            else:
                self.num_qids.append(qid)
                self.num_ranges[qid] = {
                    "max": self.org_data[qid].max(),
                    "min": self.org_data[qid].min(),
                }

    def solve_p_given_k(self, acceptance_error=1e-6):
        # Get parameter p for Retention-Replacement (for categorical qids)
        # Knowing that k monotonically decreases on p in [0,1]
        # Solve p given k using bi-section method:
        left = 0.0
        right = 1.0
        p = None
        # Init a dummy k that surely does not satisfy acceptance error
        k = self.k + acceptance_error * 10

        def __calculate_k_from_p(p):
            size_attr = [self.cat_uniques[attr].size for attr in self.cat_uniques]
            size_data = self.org_data.shape[0]
            product_list = [(1.0 - p) / (1.0 + (s - 1.0) * p) for s in size_attr]
            return 1.0 + (size_data - 1.0) * (np.prod(product_list).item() ** 2)

        while np.abs(k - self.k) > acceptance_error:
            p = (left + right) / 2.0
            k = __calculate_k_from_p(p)

            if k >= self.k:
                left = p
            else:
                right = p

        return p

    def do_retention_replacement(self):
        # Retention-Replacement
        # Perturb the categorical value based on weighted randomization:
        # The orignal value has (p + (1 - p)/attr_size) chance to remain unchanged,
        # or to be changed to one different value in the domain, each with ((1 - p)/attr_size) chance
        # - Calculate p
        # - Perturb
        p = self.solve_p_given_k()

        def __perturb(_value, _uniques, _size, _seed):
            p_all = [p + (1 - p) / _uniques.size] + [(1 - p) / _uniques.size] * (
                _uniques.size - 1
            )
            values = [_value] + [x for x in _uniques if x != _value]
            np.random.seed(_seed)
            ## Append a specific temporary "note" to the perturbed values so that
            ## they will not be mistaken as original values
            return np.random.choice(values, size=_size, p=p_all) + "#ReRe#"

        for pos, cat_qid in enumerate(self.cat_qids):
            uniques = self.cat_uniques[cat_qid]
            for index, value in enumerate(uniques):
                seed = None if self.seed is None else self.seed + int(f"1{pos}{index}")
                query = self.anon_data[cat_qid] == value
                size = self.anon_data[query].shape[0]
                self.anon_data.loc[query, cat_qid] = __perturb(
                    value, uniques, size, seed
                )

        self.anon_data.loc[:, self.cat_qids] = self.anon_data[self.cat_qids].map(
            ## Removed the temporary "note"
            lambda x: x.split("#ReRe#")[0]
        )

    def solve_b_given_k(self):
        # Get parameter b (or sigma) for Laplacian Noise (for numerical qids)
        size_data = self.org_data.shape[0]
        return 2.0 / np.log((size_data - 1.0) / (self.k - 1.0)).item()

    def do_laplacian_noise(self):
        # Laplacian Noise
        # - Calculate b (or sigma)
        # - Apply noise
        # - Truncate
        size_data = self.org_data.shape[0]
        b = self.solve_b_given_k()

        def __truncate(_x, _max_val, _min_val):
            if _x > _max_val:
                return _max_val
            if _x < _min_val:
                return _min_val
            return _x

        for pos, num_qid in enumerate(self.num_qids):
            max_val = self.num_ranges[num_qid]["max"]
            min_val = self.num_ranges[num_qid]["min"]
            b_qid = b * (max_val - min_val)
            seed = None if self.seed is None else self.seed + int(f"2{pos}")
            np.random.seed(seed)
            noise = np.random.laplace(0, b_qid, size_data)
            perturbed_data = (self.anon_data[num_qid] + noise).apply(
                lambda x: __truncate(x, max_val, min_val)
            )
            self.anon_data.loc[:, num_qid] = perturbed_data.astype(
                self.anon_data[num_qid].dtype
            )

    def anonymize(self):
        self.do_retention_replacement()
        self.do_laplacian_noise()
        self._construct_anon_data(self.anon_data, columns=list(self.anon_data))
