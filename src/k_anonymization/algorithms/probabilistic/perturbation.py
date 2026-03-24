import numpy as np

from k_anonymization.core.algorithm import Algorithm
from k_anonymization.core.dataset import Dataset


class Perturbation(Algorithm):
    """
    Implementation of Perturbation algorithm.

    Perturbation uses a Differential Privacy-inspired technique that adds
    controlled noise into the dataset. It uses "Retention-Replacement" for
    categorical attributes and "Laplacian Noise" for numerical attributes.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object holding the original data and its metadata.
    k : int
        The privacy parameter `k`.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Dataset,
        k: int,
        seed: int = None,
    ):
        """
        Initialize the Perturbation algorithm.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object holding the original data and its metadata.
        k : int
            The privacy parameter `k`.
        seed : int, optional
            Random seed for reproducibility.
        """

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
        """
        Solve for the retention parameter `p` using the bisection method.

        In the Retention-Replacement model, `k` is a function of `p`.
        Because `k` decreases monotonically as `p` increases in the
        range `[0, 1]`, this method iteratively narrows down the `p`
        required to reach the target `k`.

        Parameters
        ----------
        acceptance_error : float, default 1e-6
            The tolerance level for the difference between the
            calculated `k` and the target `k`.

        Returns
        -------
        float
            The optimal parameter `p` for categorical perturbation.
        """
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
        r"""
        Apply Retention-Replacement perturbation to categorical attributes.

        For each value, there is a probability that the original value is
        retained (:math:`p + \frac{1-p}{size}`) and a probability that it
        is replaced by another value from the domain (:math:`\frac{1-p}{size}`).

        Notes
        -----
        A temporary suffix `#ReRe#` is used during processing to distinguish
        between original and perturbed values to prevent recursive
        perturbation within the same column loop.
        """
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
        """
        Calculate the scale parameter `b` for Laplacian noise.

        This parameter determines the 'width' of the noise distribution
        needed to obscure numerical values sufficiently to reach the
        target privacy level.

        Returns
        -------
        float
            The scale parameter `b` (sigma) for the Laplace distribution.
        """
        # Get parameter b (or sigma) for Laplacian Noise (for numerical qids)
        size_data = self.org_data.shape[0]
        return (2.0 * len(self.num_qids)) / np.log(
            (size_data - 1.0) / (self.k - 1.0)
        ).item()

    def do_laplacian_noise(self):
        """
        Apply Laplacian noise to numerical attributes.

        Adds random noise sampled from a Laplace distribution centered
        at zero. The resulting values are truncated to ensure they stay
        within the original attribute's min/max range.
        """
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
        """
        Run the Perturbation algorithm.

        Applies categorical perturbation followed by numerical perturbation,
        then reconstructs the finalized anonymized data object.
        """
        self.do_retention_replacement()
        self.do_laplacian_noise()
        self._construct_anon_data(self.anon_data, columns=list(self.anon_data))
