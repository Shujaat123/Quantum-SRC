"""Offline datasets and train-only transformations."""
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def dataset(name, seed):
    loaders = {'iris': load_iris, 'wine': load_wine,
               'digits': load_digits}
    if name in loaders:
        d = loaders[name]()
        return d.data.astype(float), d.target.astype(int)
    if name != 'subspaces':
        raise ValueError('Unknown dataset: '+name)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(3):
        basis = np.linalg.qr(rng.normal(size=(12, 2)))[0]
        coeff = rng.normal(size=(120, 2))
        x = coeff @ basis.T + 0.04*rng.normal(size=(120, 12))
        xs.append(x)
        ys.extend([c]*len(x))
    return np.vstack(xs), np.array(ys)


def balanced_indices(y, per_class, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == c),
                          min(per_class, np.sum(y == c)), replace=False)
                           for c in np.unique(y)])


def prepare(name, seed, dimensions, per_class, val_per_class, test_per_class):
    X, y = dataset(name, seed)
    ids = np.arange(len(y))
    train_ids, rest = train_test_split(ids, test_size=0.4, stratify=y, random_state=seed)
    val_ids, test_ids = train_test_split(rest, test_size=0.5,
                                      stratify=y[rest], random_state=seed+1)
    rng = np.random.default_rng(seed+2)
    # Fit transformations on training partition only; dictionary is a subset.
    scaler = StandardScaler().fit(X[train_ids])
    dim = min(dimensions, X.shape[1], len(train_ids)-1)
    pca = PCA(n_components=dim, svd_solver='full').fit(scaler.transform(X[train_ids]))
    train_pca = pca.transform(scaler.transform(X[train_ids]))
    angles = MinMaxScaler(feature_range=(0, np.pi), clip=True).fit(train_pca)
    di = train_ids[balanced_indices(y[train_ids], per_class, rng)]
    vi = val_ids[balanced_indices(y[val_ids], val_per_class, rng)]
    ti = test_ids[balanced_indices(y[test_ids], test_per_class, rng)]
    def transform(i):
        return angles.transform(pca.transform(scaler.transform(X[i])))
    return {'D': transform(di), 'V': transform(vi), 'T': transform(ti),
            'yd': y[di], 'yv': y[vi], 'yt': y[ti],
            'dictionary_ids': di, 'validation_ids': vi, 'test_ids': ti,
            'training_ids': train_ids, 'pca': pca, 'scaler': scaler, 'angles': angles}
