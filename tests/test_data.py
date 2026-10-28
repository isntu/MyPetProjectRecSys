import numpy as np
import pandas as pd
import pytest

from data import load_and_split

THRESHOLD = 4
MIN_INTERACTIONS = 3


@pytest.fixture(scope='module')
def raw_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    rows = []
    for user in range(1, 61):
        items = rng.choice(np.arange(1, 16), size=12, replace=False)
        for item in items:
            rating = rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            ts = rng.integers(0, 5)
            rows.append(f"{user}::{item}::{rating}::{ts}")
    path = tmp_path_factory.mktemp('ml') / 'ratings.dat'
    path.write_text('\n'.join(rows))
    return str(path)


@pytest.fixture(scope='module')
def splits(raw_path):
    return load_and_split(
        raw_path, rel_threshold=THRESHOLD, min_interactions=MIN_INTERACTIONS
    )


def test_no_pair_overlap(splits):
    train, val, test = splits
    keys = ['user', 'item']
    assert len(train[keys].merge(val[keys])) == 0
    assert len(train[keys].merge(test[keys])) == 0
    assert len(val[keys].merge(test[keys])) == 0


def test_at_most_one_row_per_user_in_val_and_test(splits):
    _, val, test = splits
    assert val['user'].is_unique
    assert test['user'].is_unique


def test_only_positives(splits):
    for part in splits:
        assert (part['rating'] >= THRESHOLD).all()


def test_temporal_order_per_user(splits):
    train, val, test = splits
    train_max = train.groupby('user')['ts'].max().rename('train_max')
    val_ts = val.set_index('user')['ts'].rename('val_ts')
    test_ts = test.set_index('user')['ts'].rename('test_ts')

    joined = pd.concat([train_max, val_ts], axis=1, join='inner')
    assert (joined['train_max'] <= joined['val_ts']).all()

    joined = pd.concat([train_max, test_ts], axis=1, join="inner")
    assert (joined['train_max'] <= joined['test_ts']).all()

    joined = pd.concat([val_ts, test_ts], axis=1, join='inner')
    assert (joined['val_ts'] <= joined['test_ts']).all()


def test_ids_are_contiguous(splits):
    train, _, _ = splits
    assert sorted(train['user'].unique()) == list(range(train['user'].nunique()))
    assert sorted(train['item'].unique()) == list(range(train['item'].nunique()))


def test_val_test_ids_known_from_train(splits):
    train, val, test = splits
    for part in (val, test):
        assert part['user'].isin(train['user']).all()
        assert part['item'].isin(train['item']).all()
        assert not part[['user', 'item']].isna().any().any()


def test_every_user_has_train_history(splits):
    train, val, test = splits
    assert train.groupby('user').size().min() >= MIN_INTERACTIONS - 2
    for part in (val, test):
        assert part['user'].isin(train['user']).all()


def test_deterministic(raw_path):
    first = load_and_split(
        raw_path, rel_threshold=THRESHOLD, min_interactions=MIN_INTERACTIONS
    )
    second = load_and_split(
        raw_path, rel_threshold=THRESHOLD, min_interactions=MIN_INTERACTIONS
    )
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def write_rotated(path, rating_of=lambda item: 5, n_items=6):
    rows = []
    for k in range(n_items):
        for pos in range(n_items):
            item = 10 + (k + pos) % n_items
            rows.append(f'{k + 1}::{item}::{rating_of(item)}::{(pos + 1) * 100}')
    path.write_text("\n".join(rows))


def test_leave_last_out_picks_latest(tmp_path):
    path = tmp_path / 'ratings.dat'
    write_rotated(path)
    train, val, test = load_and_split(str(path), rel_threshold=4, min_interactions=2)
    assert len(test) == 6
    assert len(val) == 6
    assert (test['ts'] == 600).all()
    assert (val['ts'] == 500).all()
    assert train['ts'].max() == 400
    assert len(train) == 24


def test_tie_break_independent_of_row_order(raw_path, tmp_path):
    lines = open(raw_path).read().splitlines()
    flat = [ln.rsplit('::', 1)[0] + '::0' for ln in lines]
    rng = np.random.default_rng(1)
    shuffled = [flat[i] for i in rng.permutation(len(flat))]
    a_path = tmp_path / 'a.dat'
    b_path = tmp_path / 'b.dat'
    a_path.write_text('\n'.join(flat))
    b_path.write_text('\n'.join(shuffled))
    first = load_and_split(str(a_path), rel_threshold=THRESHOLD, min_interactions=MIN_INTERACTIONS)
    second = load_and_split(str(b_path), rel_threshold=THRESHOLD, min_interactions=MIN_INTERACTIONS)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


def test_threshold_filters_low_ratings(tmp_path):
    path = tmp_path / 'ratings.dat'
    write_rotated(path, rating_of=lambda item: 3 if item == 15 else 5)
    train, val, test = load_and_split(str(path), rel_threshold=4, min_interactions=2)
    total = pd.concat([train, val, test])
    assert (total['rating'] >= 4).all()
    assert total['item'].nunique() == 5
    assert len(total) == 6 * 5