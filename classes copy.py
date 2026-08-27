import os
import torch
from torch_geometric.transforms import BaseTransform, Compose

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as f
from torch.nn import Linear, ReLU
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch_geometric.nn import global_mean_pool
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using', device)


import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool
# meta
AVAIL_GPUS = min(1, torch.backends.mps.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GGNNModel(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        c_out,
        num_layers=3,
        dp_rate=0.1,
        use_edge_attr=False,
        edge_dim=None,
    ):
        """
        Args:
            c_in: Node input dimension
            c_hidden: Hidden dimension
            c_out: Output dimension (e.g., number of classes)
            num_layers: Number of GatedGraphConv layers
            dp_rate: Dropout rate
            use_edge_attr: Whether to include edge features
            edge_dim: Edge feature dimension (if use_edge_attr=True)
        """
        super().__init__()
        self.use_edge_attr = use_edge_attr

        self.input_lin = nn.Linear(c_in, c_hidden)

        self.ggnn_layers = nn.ModuleList([
            GatedGraphConv(out_channels=c_hidden, num_layers=1) for _ in range(num_layers)
        ])

        if self.use_edge_attr:
            assert edge_dim is not None, "edge_dim must be specified if use_edge_attr=True"
            self.edge_encoder = nn.Linear(edge_dim, c_hidden)

        self.dropout = nn.Dropout(dp_rate)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(c_hidden, c_hidden)
        self.fc2 = nn.Linear(c_hidden, c_out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if self.use_edge_attr else None
        batch = data.batch

        x = self.input_lin(x)

        for conv in self.ggnn_layers:
            if self.use_edge_attr:
                edge_emb = self.edge_encoder(edge_attr)
                x = conv(x, edge_index, edge_emb)
            else:
                x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class GraphLevelGNN(pl.LightningModule):
    def __init__(self, model_name="GGNN", **model_kwargs):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "MLP":
            self.model = MLPModel(**model_kwargs)
        else:
            self.model = GGNNModel(**model_kwargs)

        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data):
        return self.model(data)

    def shared_step(self, batch, mode="train"):
        logits = self.forward(batch)
        
        if mode == "train":
            target = batch.y[batch.train_mask] if hasattr(batch, "train_mask") else batch.y
            pred = logits[batch.train_mask] if hasattr(batch, "train_mask") else logits
        elif mode == "val":
            target = batch.y[batch.val_mask] if hasattr(batch, "val_mask") else batch.y
            pred = logits[batch.val_mask] if hasattr(batch, "val_mask") else logits
        elif mode == "test":
            target = batch.y[batch.test_mask] if hasattr(batch, "test_mask") else batch.y
            pred = logits[batch.test_mask] if hasattr(batch, "test_mask") else logits
        else:
            raise ValueError(f"Unknown mode: {mode}")

        loss = self.loss_module(pred, target)
        acc = (pred.argmax(dim=-1) == target).sum().float() / target.size(0)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, mode="train")
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.shared_step(batch, mode="val")
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        _, acc = self.shared_step(batch, mode="test")
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=5e-4)

class MLPModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """MLPModel.

        Args:
            c_in: Dimension of input features
            c_hidden: Dimension of hidden features
            c_out: Dimension of the output features. Usually number of classes in classification
            num_layers: Number of hidden layers
            dp_rate: Dropout rate to apply throughout the network

        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers - 1):
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        """Forward.

        Args:
            x: Input features per node

        """
        return self.layers(x)
    
def train_graph_regressor(model_name, dataset, target="mu", **model_kwargs):
    import os
    from torch_geometric.loader import DataLoader
    from pytorch_lightning.callbacks import ModelCheckpoint

    pl.seed_everything(42)

    # Select only the target column (e.g., mu)
    dataset.data.y = dataset.data[target].unsqueeze(1).to(torch.float)

    # Create data loader with multiple graphs per batch
    graph_data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create trainer
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel_" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
        accelerator="auto",
        devices=AVAIL_GPUS,
        max_epochs=10,
        enable_progress_bar=True,
    )
    trainer.logger._default_hp_metric = None

    # Check for pretrained model
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel_{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = GraphLevelGNN(
            model_name=model_name,
            c_in=dataset.num_node_features,
            c_out=1,  # Regression output
            **model_kwargs
        )
        trainer.fit(model, graph_data_loader, graph_data_loader)
        model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Evaluate on test graph batch
    test_result = trainer.test(model, dataloaders=graph_data_loader, verbose=False)
    batch = next(iter(graph_data_loader)).to(model.device)
    pred = model(batch)
    true = batch.y

    # Compute MAE
    mae = torch.mean(torch.abs(pred - true)).item()
    result = {"test_mae": mae}
    return model, result
