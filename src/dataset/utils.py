import random
from torch.utils.data import Subset

def build_subset(dataset, target_csv_indices, sample_frac: float = 1.0):
    """
    Create a torch.utils.data.Subset containing only the entries
    from the dataset whose "data_index" field is present in target_csv_indices.
    If sample_frac < 1.0, a random sample of this fraction is kept.
    """
    idx_map = {dataset.filtered_indices[i]: i for i in range(len(dataset))}

    subset_indices = [idx_map[csv_i] for csv_i in target_csv_indices if csv_i in idx_map]

    if 0 < sample_frac < 1.0:
        n_samples = max(1, int(len(subset_indices) * sample_frac))
        subset_indices = random.sample(subset_indices, n_samples)

    return Subset(dataset, subset_indices)
