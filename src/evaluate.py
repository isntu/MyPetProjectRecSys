import numpy as np
from scipy.sparse import csr_matrix
from metrics import recall_at_k, ndcg_at_k


def evaluate(model, train, test, k=10):
    n_users = train['user'].max() + 1
    n_items = train['item'].max() + 1
    data = np.ones(len(train), dtype=bool)
    rows = train['user']
    cols = train['item']
    train_mask = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    test_items = np.full(shape=n_users, fill_value=-1)
    test_items[test['user'].to_numpy()] = test['item'].to_numpy()
    scores = model.score(np.arange(n_users))
    metrics = {'recall': recall_at_k(scores, train_mask, test_items, k=k),
               'ndcg': ndcg_at_k(scores, train_mask, test_items, k=k)}
    return metrics