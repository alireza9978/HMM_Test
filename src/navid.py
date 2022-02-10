import numpy as np
from scipy import linalg


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
    self._init(X)
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


def _fit_log(self, X):
    log_frameprob = self._compute_log_likelihood(X)
    log_prob, fwdlattice = self._do_forward_log_pass(log_frameprob)
    bwdlattice = self._do_backward_log_pass(log_frameprob)
    posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
    return log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice


def _log_multivariate_normal_density_full(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices."""
    X_no_nan = X[~np.isnan(X)]
    X_no_nan = X_no_nan.reshape(-1, X.shape[1])

    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob
