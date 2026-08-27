from rdkit import Chem
from torch_geometric.transforms import BaseTransform, Compose
import torch
from torch_geometric.datasets import QM9

class FeatureAblation(BaseTransform):
    def __init__(self, keep_indices):
        super().__init__()
        self.keep_indices = keep_indices

    def __call__(self, data):
        # Keep only selected features
        data.x = data.x[:, self.keep_indices]
        return data    # return data

class SetTarget(BaseTransform):
    '''
    Transform to modify target vector so there is a single value.

    Choose from 0-18 for target values described here:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9
    '''
    def __init__(self,target):
        self.target = target

    def __call__(self,data):
        data.y = data.y[:, self.target]
        return data
    
class NormalizeTarget(BaseTransform):
    '''
    Transform to normalize target vector.

    Parameters:
    ----------
    mean : mean of the training data target values
    std : standard deviation of the training data target values

    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data.y - self.mean) / self.std
        return data


def load_qm9_with_ablation(keep_indices, target_idx=5, num_samples=20_000):
    # Step 1: Load raw dataset and restrict size
    base_dataset = QM9(root="data")
    base_dataset.data.y = base_dataset.data.y[:, target_idx]
    base_dataset = base_dataset[:num_samples]  # Limit to first 20k samples

    # Step 2: Train/val/test split on that subset
    train_len = int(0.8 * num_samples)
    val_len = int(0.1 * num_samples)
    test_len = num_samples - train_len - val_len

    train_set, val_set, test_set = torch.utils.data.random_split(
        base_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )

    # Step 3: Compute normalization stats from train split only
    y_train = torch.cat([data.y for data in train_set])
    target_mean = y_train.mean().item()
    target_std = y_train.std().item()

    # Step 4: Reload same data subset with ablation and normalization transforms
    full_transformed = QM9(
        root="data",
        transform=Compose([
            FeatureAblation(keep_indices),
            NormalizeTarget(mean=target_mean, std=target_std)
        ])
    )
    full_transformed.data.y = full_transformed.data.y[:, target_idx]
    full_transformed = full_transformed[:num_samples]  # Limit again

    return full_transformed, target_mean, target_std
