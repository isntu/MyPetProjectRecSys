import numpy as np
from implicit.cpu.bpr import BayesianPersonalizedRanking

from data import build_matrix


class BPR:
    def __init__(self, n_factors=64, learning_rate=0.05, reg=0.01, n_iter=100, seed=0):
        self.model = BayesianPersonalizedRanking(
            factors=n_factors,
            learning_rate=learning_rate,
            regularization=reg,
            iterations=n_iter,
            num_threads=1,
            random_state=seed,
        )

    def fit(self, train):
        n_users = train['user'].max() + 1
        n_items = train['item'].max() + 1
        x = build_matrix(train, n_users, n_items).tocsr()
        self.model.fit(x, show_progress=False)
        return self

    def score(self, users):
        user_factors = np.asarray(self.model.user_factors)
        item_factors = np.asarray(self.model.item_factors)
        return user_factors[users] @ item_factors.T