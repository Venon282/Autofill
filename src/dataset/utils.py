"""Dataset helper utilities."""

from torch.utils.data import Subset


def build_subset(dataset, target_csv_indices):
    """Return a :class:`Subset` containing only the entries matching ``target_csv_indices``."""

    idx_map = {dataset.csv_index[i]: i for i in range(len(dataset))}
    subset_indices = [idx_map[csv_i] for csv_i in target_csv_indices if csv_i in idx_map]
    return Subset(dataset, subset_indices)
