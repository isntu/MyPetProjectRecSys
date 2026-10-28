import numpy as np


class Popularity:
    def __init__(self):
        pass
    
    def fit(self, train):
        self.n_items = train['item'].max() + 1
        self.counts = train.groupby('item').size().reindex(range(self.n_items), fill_value=0).to_numpy()
        return self

    def score(self, users):
        return np.tile(self.counts, (len(users), 1))
