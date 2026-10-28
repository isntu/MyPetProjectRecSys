import numpy as np

from data import build_matrix


class IALS:
    def __init__(self, n_factors=64, alpha=10.0, reg=0.1, n_iter=15, seed=0):
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg = reg
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def _update(self, matrix, fixed):
        gram = fixed.T @ fixed
        ridge = self.reg * np.eye(self.n_factors)
        result = np.zeros((matrix.shape[0], self.n_factors))
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            y = fixed[matrix.indices[start:end]]
            a = gram + ridge + self.alpha * (y.T @ y)
            b = (1.0 + self.alpha) * y.sum(axis=0)
            result[row] = np.linalg.solve(a, b)
        return result

    def fit(self, train):
        n_users = train['user'].max() + 1
        n_items = train['item'].max() + 1
        x = build_matrix(train, n_users, n_items)
        xt = x.T.tocsr()
        scale = 0.01
        self.P = self.rng.normal(0, scale, (n_users, self.n_factors))
        self.Q = self.rng.normal(0, scale, (n_items, self.n_factors))
        for _ in range(self.n_iter):
            self.P = self._update(x, self.Q)
            self.Q = self._update(xt, self.P)
        return self

    def score(self, users):
        return self.P[users] @ self.Q.T