import numpy as np
import pandas as pd
import pytest

from data import build_matrix
from models.ials import IALS


N_USERS = 30
N_ITEMS = 20


@pytest.fixture(scope='module')
def train():
    rng = np.random.default_rng(0)
    rows = []
    for user in range(N_USERS):
        items = rng.choice(N_ITEMS, size=rng.integers(3, 10), replace=False)
        rows.extend((user, item) for item in items)
    frame = pd.DataFrame(rows, columns=['user', 'item'])
    frame.loc[0, 'user'] = 0
    return frame


def loss(p, q, x_dense, alpha, reg):
    confidence = 1.0 + alpha * x_dense
    residual = x_dense - p @ q.T
    return (confidence * residual**2).sum() + reg * ((p**2).sum() + (q**2).sum())


def test_loss_never_increases(train):
    model = IALS(n_factors=4, alpha=5.0, reg=0.1)
    x = build_matrix(train, N_USERS, N_ITEMS)
    xt = x.T.tocsr()
    dense = x.toarray()
    rng = np.random.default_rng(1)
    p = rng.normal(0, 0.1, (N_USERS, 4))
    q = rng.normal(0, 0.1, (N_ITEMS, 4))
    history = [loss(p, q, dense, 5.0, 0.1)]
    for _ in range(6):
        p = model._update(x, q)
        history.append(loss(p, q, dense, 5.0, 0.1))
        q = model._update(xt, p)
        history.append(loss(p, q, dense, 5.0, 0.1))
    assert np.all(np.diff(history) <= 1e-9)
    assert history[-1] < history[0]


def test_update_matches_direct_formula(train):
    alpha, reg = 7.0, 0.3
    model = IALS(n_factors=3, alpha=alpha, reg=reg)
    x = build_matrix(train, N_USERS, N_ITEMS)
    dense = x.toarray()
    rng = np.random.default_rng(2)
    q = rng.normal(size=(N_ITEMS, 3))
    fast = model._update(x, q)
    for u in range(N_USERS):
        c = np.diag(1.0 + alpha * dense[u])
        a = q.T @ c @ q + reg * np.eye(3)
        b = q.T @ c @ dense[u]
        assert np.allclose(fast[u], np.linalg.solve(a, b))


def test_score_shape_and_determinism(train):
    first = IALS(n_factors=4, n_iter=3, seed=5).fit(train)
    second = IALS(n_factors=4, n_iter=3, seed=5).fit(train)
    users = np.arange(N_USERS)
    scores = first.score(users)
    assert scores.shape == (N_USERS, N_ITEMS)
    assert np.allclose(scores, second.score(users))


def test_user_without_items_gets_zero_vector():
    train = pd.DataFrame({'user': [0, 0, 2, 2], 'item': [0, 1, 1, 2]})
    model = IALS(n_factors=2, n_iter=2).fit(train)
    assert np.allclose(model.P[1], 0.0)