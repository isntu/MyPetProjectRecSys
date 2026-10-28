import numpy as np
import pytest
from scipy.sparse import csr_matrix

from metrics import ndcg_at_k, recall_at_k


def empty_mask(n_users, n_items):
    return csr_matrix((n_users, n_items), dtype=bool)


def mask_from(pairs, n_users, n_items):
    if not pairs:
        return empty_mask(n_users, n_items)
    rows, cols = zip(*pairs)
    data = np.ones(len(pairs), dtype=bool)
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def single_user(scores_row, test_item, k, masked=()):
    scores = np.array([scores_row], dtype=float)
    mask = mask_from([(0, j) for j in masked], 1, scores.shape[1])
    test_items = np.array([test_item])
    return (
        recall_at_k(scores, mask, test_items, k),
        ndcg_at_k(scores, mask, test_items, k),
    )


def test_perfect_ranking_gives_one():
    recall, ndcg = single_user([0.1, 0.9, 0.5, 0.3], test_item=1, k=3)
    assert recall == pytest.approx(1.0)
    assert ndcg == pytest.approx(1.0)


def test_miss_gives_zero():
    recall, ndcg = single_user([0.9, 0.1, 0.8, 0.7], test_item=1, k=2)
    assert recall == 0.0
    assert ndcg == 0.0


def test_rank_three_inside_k():
    recall, ndcg = single_user([0.9, 0.8, 0.7, 0.1], test_item=2, k=3)
    assert recall == pytest.approx(1.0)
    assert ndcg == pytest.approx(1.0 / np.log2(4))


def test_rank_three_outside_k():
    recall, ndcg = single_user([0.9, 0.8, 0.7, 0.1], test_item=2, k=2)
    assert recall == 0.0
    assert ndcg == 0.0


def test_rank_two_value():
    _, ndcg = single_user([0.9, 0.8, 0.7, 0.1], test_item=1, k=4)
    assert ndcg == pytest.approx(1.0 / np.log2(3))


def test_train_items_are_masked():
    scores_row = [0.99, 0.8, 0.5, 0.1]
    recall_masked, ndcg_masked = single_user(scores_row, test_item=1, k=1, masked=[0])
    recall_open, ndcg_open = single_user(scores_row, test_item=1, k=1)
    assert recall_masked == pytest.approx(1.0)
    assert ndcg_masked == pytest.approx(1.0)
    assert recall_open == 0.0
    assert ndcg_open == 0.0


def test_masking_improves_rank_value():
    scores_row = [0.99, 0.8, 0.5, 0.1]
    _, ndcg_masked = single_user(scores_row, test_item=2, k=4, masked=[0])
    _, ndcg_open = single_user(scores_row, test_item=2, k=4)
    assert ndcg_masked == pytest.approx(1.0 / np.log2(3))
    assert ndcg_open == pytest.approx(1.0 / np.log2(4))


def test_averaging_over_users():
    scores = np.array(
        [
            [0.9, 0.1, 0.2, 0.3],
            [0.9, 0.8, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.9],
        ]
    )
    test_items = np.array([0, 1, 0])
    mask = empty_mask(3, 4)
    expected_ndcg = (1.0 + 1.0 / np.log2(3) + 0.0) / 3
    assert recall_at_k(scores, mask, test_items, 2) == pytest.approx(2 / 3)
    assert ndcg_at_k(scores, mask, test_items, 2) == pytest.approx(expected_ndcg)


def test_users_without_test_item_are_skipped():
    scores = np.array(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.9, 0.2],
            [0.1, 0.2, 0.9],
        ]
    )
    test_items = np.array([0, -1, 1])
    mask = empty_mask(3, 3)
    assert recall_at_k(scores, mask, test_items, 1) == pytest.approx(0.5)
    assert ndcg_at_k(scores, mask, test_items, 1) == pytest.approx(0.5)


def test_ties_are_resolved_pessimistically():
    scores = np.zeros((1, 10))
    mask = empty_mask(1, 10)
    test_items = np.array([4])
    assert recall_at_k(scores, mask, test_items, 3) == 0.0
    assert ndcg_at_k(scores, mask, test_items, 3) == 0.0


def test_ties_only_among_equal_scores():
    scores = np.array([[0.9, 0.5, 0.5, 0.1]])
    mask = empty_mask(1, 4)
    test_items = np.array([1])
    assert recall_at_k(scores, mask, test_items, 2) == 0.0
    assert recall_at_k(scores, mask, test_items, 3) == pytest.approx(1.0)
    assert ndcg_at_k(scores, mask, test_items, 3) == pytest.approx(1.0 / np.log2(4))


def test_k_larger_than_catalog():
    recall, ndcg = single_user([0.1, 0.9, 0.5], test_item=2, k=50)
    assert recall == pytest.approx(1.0)
    assert ndcg == pytest.approx(1.0 / np.log2(3))


def test_inputs_are_not_mutated():
    scores = np.array([[0.9, 0.8, 0.7, 0.1], [0.1, 0.2, 0.3, 0.4]])
    mask = mask_from([(0, 0), (1, 3)], 2, 4)
    test_items = np.array([1, 2])
    scores_copy = scores.copy()
    test_items_copy = test_items.copy()
    mask_copy = mask.copy()
    recall_at_k(scores, mask, test_items, 2)
    ndcg_at_k(scores, mask, test_items, 2)
    assert np.array_equal(scores, scores_copy)
    assert np.array_equal(test_items, test_items_copy)
    assert (mask != mask_copy).nnz == 0


@pytest.mark.parametrize('k', [1, 5, 10, 40])
def test_matches_brute_force_on_random_data(k):
    rng = np.random.default_rng(42)
    n_users, n_items = 50, 40
    scores = rng.normal(size=(n_users, n_items))
    dense_mask = rng.random((n_users, n_items)) < 0.3
    test_items = np.empty(n_users, dtype=int)
    for u in range(n_users):
        free = np.flatnonzero(~dense_mask[u])
        test_items[u] = rng.choice(free)
    test_items[::7] = -1
    mask = csr_matrix(dense_mask)

    hits, gains, count = 0.0, 0.0, 0
    for u in range(n_users):
        if test_items[u] < 0:
            continue
        row = np.where(dense_mask[u], -np.inf, scores[u])
        order = np.argsort(-row, kind='stable')[:k]
        count += 1
        if test_items[u] in order:
            position = int(np.flatnonzero(order == test_items[u])[0]) + 1
            hits += 1.0
            gains += 1.0 / np.log2(position + 1)

    assert recall_at_k(scores, mask, test_items, k) == pytest.approx(hits / count)
    assert ndcg_at_k(scores, mask, test_items, k) == pytest.approx(gains / count)


def test_metrics_are_in_unit_interval_and_ndcg_le_recall():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(30, 25))
    mask = empty_mask(30, 25)
    test_items = rng.integers(0, 25, size=30)
    recall = recall_at_k(scores, mask, test_items, 5)
    ndcg = ndcg_at_k(scores, mask, test_items, 5)
    assert 0.0 <= ndcg <= recall <= 1.0