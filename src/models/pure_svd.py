from scipy.sparse.linalg import svds

from data import build_matrix

class PureSVD:
    def __init__(self, n_factors=64):
        self.n_factors = n_factors

    def fit(self, train):
        n_users = train['user'].max() + 1
        n_items = train['item'].max() + 1
        self.X = build_matrix(train, n_users, n_items)
        _, _, vt = svds(self.X, k=self.n_factors)
        self.V = vt.T
        return self

    def score(self, users):
        return (self.X[users] @ self.V) @ self.V.T
