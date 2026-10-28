import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix



def load_and_split(path, rel_threshold=4, min_interactions=5):
    df = pd.read_csv(
        path,
        sep='::' ,
        engine='python',
        names=['user', 'item', 'rating', 'ts'],
    )
    df = df[df['rating'] >= rel_threshold]

    item_size = df.groupby('item')['user'].transform('size')
    df = df[item_size >= min_interactions]

    user_size = df.groupby('user')[ 'item'].transform('size')
    df = df[user_size >= min_interactions]

    df = df.sort_values(['user', 'ts' , 'item']).reset_index(drop=True)

    rank_from_end = df.groupby('user').cumcount(ascending=False)
    train = df[rank_from_end >=  2].copy()
    val = df[rank_from_end == 1].copy()
    test = df[rank_from_end == 0].copy()

    train_items = set(train['item'])
    val = val[val['item'].isin(train_items)]
    test = test[test['item'].isin(train_items)]

    user_map = {u: i for i, u in enumerate(sorted(train["user"].unique()))}
    item_map = {m: i for i, m in enumerate(sorted(train_items))}

    out = []
    for part in (train, val, test):
        part = part.copy()
        part['user'] = part['user'].map(user_map)
        part['item'] = part['item'].map(item_map)
        out.append(part.reset_index(drop=True))

    return tuple(out)

def build_matrix(train, n_users, n_items):
    data = np.ones(len(train))
    rows = train['user']
    cols = train['item']
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
