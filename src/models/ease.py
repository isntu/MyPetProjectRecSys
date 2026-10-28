import numpy as np

from data import build_matrix


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, train):
        n_users = train['user'].max() + 1
        n_items = train['item'].max() + 1
        self.X = build_matrix(train, n_users, n_items)
        gram = (self.X.T @ self.X).toarray()
        gram[np.diag_indices_from(gram)] += self.reg
        p = np.linalg.inv(gram)
        b = p / (-np.diag(p))
        np.fill_diagonal(b, 0.0)
        self.B = b
        return self

    def score(self, users):
        return self.X[users] @ self.B