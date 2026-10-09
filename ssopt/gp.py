"""Minimal multi-output Gaussian process (shared anisotropic RBF kernel).

numpy/scipy only — OSC's python/3.12 module has no scikit-learn.

Model:  y_j(x) ~ GP(0, k),  k(x,x') = a² exp(-½ Σ_i (x_i-x'_i)²/ℓ_i²) + σ² δ
All outputs share (a, ℓ, σ); hyperparameters maximise the summed log marginal
likelihood. Inputs are expected in [0,1]^d, outputs standardised.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class MultiOutputGP:
    def __init__(self, length_scale=0.7, amplitude=1.0, noise=1e-3,
                 ls_bounds=(0.05, 50.0), noise_bounds=(1e-6, 1.0), amp_bounds=(1e-2, 1e2),
                 n_restarts: int = 1, seed: int = 0):
        self.ls0, self.amp0, self.noise0 = length_scale, amplitude, noise
        self.ls_bounds, self.noise_bounds, self.amp_bounds = ls_bounds, noise_bounds, amp_bounds
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)

    # --- kernel -------------------------------------------------------------
    @staticmethod
    def _k(X1, X2, amp, ls):
        d = (X1[:, None, :] - X2[None, :, :]) / ls
        return amp ** 2 * np.exp(-0.5 * np.sum(d * d, axis=-1))

    def _unpack(self, theta):
        d = self.X.shape[1]
        return np.exp(theta[0]), np.exp(theta[1:1 + d]), np.exp(theta[1 + d])

    def _nlml(self, theta):
        amp, ls, noise = self._unpack(theta)
        K = self._k(self.X, self.X, amp, ls) + (noise ** 2 + 1e-10) * np.eye(len(self.X))
        try:
            c, low = cho_factor(K, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve((c, low), self.Y, check_finite=False)
        n, m = self.Y.shape
        logdet = 2.0 * np.sum(np.log(np.diag(c)))
        return 0.5 * np.sum(self.Y * alpha) + 0.5 * m * logdet + 0.5 * n * m * np.log(2 * np.pi)

    # --- fit / predict --------------------------------------------------------
    def fit(self, X, Y):
        self.X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        self.Y = Y[:, None] if Y.ndim == 1 else Y
        n, d = self.X.shape
        lb = np.concatenate([[np.log(self.amp_bounds[0])], np.full(d, np.log(self.ls_bounds[0])),
                             [np.log(self.noise_bounds[0])]])
        ub = np.concatenate([[np.log(self.amp_bounds[1])], np.full(d, np.log(self.ls_bounds[1])),
                             [np.log(self.noise_bounds[1])]])
        starts = [np.concatenate([[np.log(self.amp0)], np.full(d, np.log(self.ls0)), [np.log(self.noise0)]])]
        for _ in range(self.n_restarts):
            starts.append(lb + self.rng.random(len(lb)) * (ub - lb))
        best = None
        for t0 in starts:
            r = minimize(self._nlml, t0, method="L-BFGS-B", bounds=list(zip(lb, ub)),
                         options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        amp, ls, noise = self._unpack(self.theta)
        self.amp, self.ls, self.noise = amp, ls, noise
        K = self._k(self.X, self.X, amp, ls) + (noise ** 2 + 1e-10) * np.eye(n)
        self._chol = cho_factor(K, lower=True, check_finite=False)
        self._alpha = cho_solve(self._chol, self.Y, check_finite=False)
        return self

    def fit_fixed(self, X, Y, theta):
        """Condition on (X, Y) with given hyperparameters — no optimisation (cheap refits, CV folds)."""
        self.X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        self.Y = Y[:, None] if Y.ndim == 1 else Y
        self.theta = np.asarray(theta, float)
        self.amp, self.ls, self.noise = self._unpack(self.theta)
        K = self._k(self.X, self.X, self.amp, self.ls) + (self.noise ** 2 + 1e-10) * np.eye(len(self.X))
        self._chol = cho_factor(K, lower=True, check_finite=False)
        self._alpha = cho_solve(self._chol, self.Y, check_finite=False)
        return self

    def predict(self, Xs, return_std: bool = False):
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = self._k(Xs, self.X, self.amp, self.ls)
        mu = Ks @ self._alpha
        if not return_std:
            return mu
        v = cho_solve(self._chol, Ks.T, check_finite=False)
        var = self.amp ** 2 - np.sum(Ks * v.T, axis=1)
        sd = np.sqrt(np.maximum(var, 1e-12))
        return mu, np.repeat(sd[:, None], mu.shape[1], axis=1)

    @property
    def kernel_(self):
        return f"amp={self.amp:.3g} ls={np.round(self.ls, 3).tolist()} noise={self.noise:.2g}"
