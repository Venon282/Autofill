"""Dataset helper utilities."""

from torch.utils.data import Subset

def build_subset(dataset, target_csv_indices):
    """
    Crée un torch.utils.data.Subset contenant uniquement les entrées
    du dataset dont le champ "data_index" figure dans target_csv_indices.
    """

    # On crée une table de correspondance entre le data_index et l'indice interne du dataset
    idx_map = {dataset[i]["data_index"]: i for i in range(len(dataset))}

    # On garde uniquement les indices internes correspondant aux data_index demandés
    subset_indices = [idx_map[csv_i] for csv_i in target_csv_indices if csv_i in idx_map]

    # On crée le Subset PyTorch
    return Subset(dataset, subset_indices)