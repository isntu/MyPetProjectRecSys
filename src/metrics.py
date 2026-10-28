import numpy as np


def _ranks(scores, train_mask, test_items):
    test_items = np.asarray(test_items)
    users = np.flatnonzero(test_items >= 0)
    if users.size == 0:
        raise ValueError('no users with a test item')

    block = np.array(scores[users], dtype=float)
    rows, cols = train_mask[users].nonzero()
    block[rows, cols] = -np.inf

    target = block[np.arange(users.size), test_items[users]]
    return (block >= target[:, None]).sum(axis=1)


def recall_at_k(scores, train_mask, test_items, k):
    ranks = _ranks(scores, train_mask, test_items)
    return float((ranks <= k).mean())


def ndcg_at_k(scores, train_mask, test_items, k):
    ranks = _ranks(scores, train_mask, test_items)
    gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return float(gains.mean())