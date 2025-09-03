from torch.utils.data import Subset

def build_subset(dataset, target_csv_indices):
    # On récupère la map csv_index -> index interne du dataset
    idx_map = {dataset.csv_index[i]: i for i in range(len(dataset))}
    
    # On sélectionne uniquement les indices présents dans la liste donnée
    subset_indices = [idx_map[csv_i] for csv_i in target_csv_indices if csv_i in idx_map]
    
    return Subset(dataset, subset_indices)
