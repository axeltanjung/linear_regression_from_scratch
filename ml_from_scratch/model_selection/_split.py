import numpy as np

class KFold:
    '''
    K-Fold cross-validator

    Provide train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by)

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 5

    shuffle : bool, default=False
        Wheater shuffle the data before splitting into batches

    random_state : int, default=42
        When 'shuffle' is True, 'random_state' affects the ordering
        of the indices. Otherwise, this parameter has no affect
    '''
    def __init__(
        self,
        n_splits=5,
        shuffle=False,
        random_state=42
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X):
        # cari jumlah data yang ingin di split
        n_samples = len(X)
        indices = np.arange(n_samples)
        indices

        # Randomize data
        if self.shuffle:
            # Set random seed
            np.random.seed(self.random_state)

            # Shuffle data
            np.random.shuffle(indices)

        # Tentukan jumlah split
        n_splits = self.n_splits
        fold_sizes = np.ones(n_splits, dtype=int) * int(n_samples / n_splits)
        fold_sizes[: n_samples%n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            end = fold_size + current
            
            # Seleksi index untuk data validasi
            yield indices[start:end]

            # Update current index
            current = end

    def split(self, X):
        # Inisiasi
        n_samples = len(X)
        indices = np.arange(n_samples)

        for test_index in self._iter_test_indices(X):
            train_index = np.array([ind for ind in indices if ind not in test_index])

            yield (train_index, test_index)
