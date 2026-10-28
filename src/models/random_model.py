import numpy as np


class RandomModel:
    def __init__(self, random_state=21):
        self.rng = np.random.default_rng(random_state)

    def fit(self, train):
        self.n_items = train['item'].max() + 1
        return self

    def score(self, n_users):
        return self.rng.random((len(n_users), self.n_items))