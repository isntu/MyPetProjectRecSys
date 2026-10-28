import math

import numpy as np

from data import build_matrix
from metrics import _ranks


def per_user_metrics(model, train, test, k=10):
    n_users = train["user"].max() + 1
    n_items = train["item"].max() + 1
    train_mask = build_matrix(train, n_users, n_items).astype(bool)
    test_items = np.full(n_users, -1)
    test_items[test["user"].to_numpy()] = test["item"].to_numpy()
    scores = model.score(np.arange(n_users))
    ranks = _ranks(scores, train_mask, test_items)
    hit = ranks <= k
    recall = hit.astype(float)
    ndcg = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)
    return recall, ndcg


def paired_difference(a, b):
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / math.sqrt(n)
    if se == 0:
        z, p_value = 0.0, 1.0
    else:
        z = mean / se
        p_value = math.erfc(abs(z) / math.sqrt(2))
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff": float(mean),
        "se": float(se),
        "ci95": (float(mean - 1.96 * se), float(mean + 1.96 * se)),
        "z": float(z),
        "p_value": float(p_value),
        "n_users": n,
    }