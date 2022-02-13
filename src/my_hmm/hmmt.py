from hmmlearn.hmm import GaussianHMM, _check_and_set_gaussian_n_features
from hmmlearn import _utils
import logging
from sklearn import cluster
import numpy as np

from sklearn.utils import check_array

_log = logging.getLogger(__name__)


class GaussianHMMT(GaussianHMM):
    """
    Hidden Markov Model with Gaussian emissions.
    but input data can contain missing values
    """

    def _init_with_nan(self, X):
        X_no_nan = X[~np.isnan(X)]
        X_no_nan = X_no_nan.reshape(-1, X.shape[1])

        _check_and_set_gaussian_n_features(self, X_no_nan)
        super()._init(X_no_nan)

        if self._needs_init("m", "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X_no_nan)
            self.means_ = kmeans.cluster_centers_
        if self._needs_init("c", "covars_"):
            cv = np.cov(X_no_nan.T) + self.min_covar * np.eye(X_no_nan.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

    def fit_wit_nan(self, X, lengths=None):
        """
        Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, force_all_finite="allow-nan")
        self._init_with_nan(X)
        self._check()

        self.monitor_._reset()

        impl = {
            "scaling": self._fit_scaling,
            "log": self._fit_log,
        }[self.implementation]
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_log_prob = 0
            for sub_X in _utils.split_X_lengths(X, lengths):
                lattice, log_prob, posteriors, fwdlattice, bwdlattice = impl(sub_X)
                # Derived HMM classes will implement the following method to
                # update their probability distributions, so keep
                # a single call to this method for simplicity.
                self._accumulate_sufficient_statistics(
                    stats, sub_X, lattice, posteriors, fwdlattice,
                    bwdlattice)
                curr_log_prob += log_prob

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            self.monitor_.report(curr_log_prob)
            if self.monitor_.converged:
                break

        if (self.transmat_.sum(axis=1) == 0).any():
            _log.warning("Some rows of transmat_ have zero sum because no "
                         "transition from the state was ever observed.")

        return self
