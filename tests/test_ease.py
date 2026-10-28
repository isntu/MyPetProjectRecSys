import numpy as np
import pandas as pd
import pytest

from data import build_matrix
from models.ease import EASE


N_USERS = 40
N_ITEMS = 12


@pytest.fixture(scope='module')
def train():
    rng = np.random.default_rng(0)
    rows = []
    for user in range(N_USERS):
        items = rng.choice(N_ITEMS, size=rng.integers(3, 8), replace=False)
        rows.extend((user, item) for item in items)
    return pd.DataFrame(rows, columns=['user', 'item'])


def test_diagonal_is_zero(train):
    model = EASE(reg=2.0).fit(train)
    assert np.all(np.diag(model.B) == 0.0)
    assert model.B.shape == (N_ITEMS, N_ITEMS)


def test_columns_are_ridge_regressions(train):
    reg = 3.0
    model = EASE(reg=reg).fit(train)
    x = build_matrix(train, N_USERS, N_ITEMS).toarray()
    for j in range(N_ITEMS):
        others = [i for i in range(N_ITEMS) if i != j]
        xo = x[:, others]
        coef = np.linalg.solve(xo.T @ xo + reg * np.eye(len(others)), xo.T @ x[:, j])
        assert np.allclose(model.B[others, j], coef)


def test_score_shape_and_values(train):
    model = EASE(reg=2.0).fit(train)
    users = np.array([0, 5, 7])
    scores = model.score(users)
    x = build_matrix(train, N_USERS, N_ITEMS).toarray()
    assert scores.shape == (3, N_ITEMS)
    assert np.allclose(scores, x[users] @ model.B)


def test_larger_reg_shrinks_weights(train):
    small = EASE(reg=1.0).fit(train)
    large = EASE(reg=1000.0).fit(train)
    assert np.abs(large.B).sum() < np.abs(small.B).sum()