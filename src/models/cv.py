"""
PurgedTimeSeriesSplit — López de Prado (2018), Advances in Financial ML.
Évite le leakage entre folds adjacents via purge + embargo.
"""

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Walk-forward CV avec purge et embargo pour séries financières.

    Purge  : retire du train les observations dont le label chevauche la fenêtre test.
    Embargo: retire les N observations immédiatement après le test (autocorrélation résiduelle).

    Parameters
    ----------
    n_splits    : nombre de folds
    embargo_pct : fraction des données utilisée comme embargo après chaque fold test
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        embargo = int(n * self.embargo_pct)

        for i in range(1, self.n_splits + 1):
            test_start = i * fold_size
            test_end = test_start + fold_size
            purge_start = max(0, test_start - embargo)
            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(min(test_end + embargo, n), n),
            ])
            test_idx = np.arange(test_start, min(test_end, n))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
