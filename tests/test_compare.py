import numpy as np
import pandas as pd
import pytest

from compare import paired_difference, per_user_metrics
from metrics import ndcg_at_k, recall_at_k
from data import build_matrix


class FixedScores:
    def __init__(self, scores):
        self.scores = scores

    def score(self, users):
        return self.scores[users]


@pytest.fixture(scope='module')
def setup():
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 25
    rows = []
    for user in range(n_users):
        items = rng.choice(n_items, size=6, replace=False)
        rows.extend((user, item) for item in items[:4])
    train = pd.DataFrame(rows, columns=['user', 'item'])
    test_rows = []
    for user in range(0, n_users, 2):
        seen = set(train.loc[train['user'] == user, 'item'])
        free = [i for i in range(n_items) if i not in seen]
        test_rows.append((user, rng.choice(free)))
    test = pd.DataFrame(test_rows, columns=['user', 'item'])
    scores = rng.normal(size=(n_users, n_items))
    return train, test, scores, n_users, n_items


def test_per_user_means_match_aggregate_metrics(setup):
    train, test, scores, n_users, n_items = setup
    recall, ndcg = per_user_metrics(FixedScores(scores), train, test, k=5)
    mask = build_matrix(train, n_users, n_items).astype(bool)
    test_items = np.full(n_users, -1)
    test_items[test['user'].to_numpy()] = test['item'].to_numpy()
    assert recall.mean() == pytest.approx(recall_at_k(scores, mask, test_items, 5))
    assert ndcg.mean() == pytest.approx(ndcg_at_k(scores, mask, test_items, 5))
    assert len(recall) == len(test)


def test_identical_vectors_give_zero_difference():
    a = np.array([1.0, 0.0, 1.0, 0.0])
    result = paired_difference(a, a)
    assert result['diff'] == 0.0
    assert result['p_value'] == 1.0


def test_known_paired_difference():
    a = np.array([1.0, 1.0, 1.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    result = paired_difference(a, b)
    assert result['diff'] == pytest.approx(0.5)
    assert result['se'] == pytest.approx(np.std([1, 0, 1, 0], ddof=1) / 2)
    low, high = result['ci95']
    assert low < 0.5 < high


def test_clear_winner_is_significant():
    rng = np.random.default_rng(1)
    b = (rng.random(5000) < 0.05).astype(float)
    a = np.maximum(b, (rng.random(5000) < 0.05).astype(float))
    assert paired_difference(a, b)['p_value'] < 0.01