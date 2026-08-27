import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#pyG
import torch_geometric
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint

# torch-geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, Set2Set
from torch_geometric.transforms import BaseTransform, Compose

#config
from config import get_gnn_output_dim, gnn_layer_by_name, CHEMICAL_ACCURACY_MU, model_path
from tower import EdgeMLP, ToweredGNNModel

# meta

import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    ACCELERATOR = "mps"
    AVAIL_GPUS = 1  # Still set this to 1 for compatibility with the trainer
    print("✅ Using Apple MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    ACCELERATOR = "cpu"
    AVAIL_GPUS = 0
    print("⚠️ MPS not available, falling back to CPU")

BATCH_SIZE = 256 if AVAIL_GPUS else 64
# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "models/")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class EdgeAwareGGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, use_edge_attr=False):
        super().__init__(aggr='add')  # or 'mean' / 'max'
        self.use_edge_attr = use_edge_attr
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        if self.use_edge_attr:
            self.edge_encoder = nn.Linear(hidden_dim, hidden_dim)

        self.lin = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            edge_attr = None

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x = self.gru(out, x)
        return x

    def message(self, x_j, edge_attr):
        if self.use_edge_attr and edge_attr is not None:
            return x_j + edge_attr
        else:
            return x_j

class GNNModel(nn.Module):
    def __init__(
        self,
        n_input_dim,
        n_hidden_features,
        n_output_dim,
        num_layers=3,
        dropout=0.1,
        edge_attr_dim=None,
        use_edge_attr = True,
        **kwargs
    ):
        """
        GGNN model using GatedGraphConv from torch_geometric.
        """
        super().__init__()

        assert edge_attr_dim is not None, "GGNN requires edge features"

        self.use_edge_attr = use_edge_attr

        if use_edge_attr:
            print('use edge attribute')
            assert edge_attr_dim is not None, "Edge features are required if use_edge_attr=True"
            self.edge_encoder = nn.Linear(edge_attr_dim, n_hidden_features)

        self.ggnn = geom_nn.GatedGraphConv(
            out_channels=n_hidden_features,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.project_in = nn.Linear(n_input_dim, n_hidden_features)

           # Stack of GGNN layers
        self.layers = nn.ModuleList([
            EdgeAwareGGNNLayer(n_hidden_features, use_edge_attr=use_edge_attr)
            for _ in range(num_layers)
        ])

        # Readout head
        self.readout = nn.Sequential(
            nn.Linear(n_hidden_features, n_hidden_features//2),
            nn.ReLU(),
            nn.Linear(n_hidden_features//2, 1)
        )
        self.pool = global_mean_pool
        #self.pool = Set2Set(n_hidden_features, 3)

    # def forward(self, x, edge_index, batch, edge_attr=None):
    #     x = self.project_in(x)
    #     x = self.ggnn(x, edge_index)
    #     x = self.relu(x)
    #     x = self.dropout(x)
    #     x = self.pool(x, batch)
    #     return self.readout(x)
    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.project_in(x)

        if self.use_edge_attr:
            assert edge_attr is not None, "Edge features must be provided if use_edge_attr=True"
            edge_attr = self.edge_encoder(edge_attr)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.pool(x, batch)
        return self.readout(x).view(-1)
    
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

             
class GraphLevelGNN(pl.LightningModule):

    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # normalization scalers
        self.target_mean = kwargs.pop("target_mean", None)
        self.target_std = kwargs.pop("target_std", None)

        if model_name == "TowerGNN":
            self.model = ToweredGNNModel(**kwargs)
        elif model_name == "MLP":
            self.model = MLPModel(**kwargs)
        else:
            self.model = GNNModel( **kwargs)

        self.loss_module = nn.MSELoss()
        self.mae_metric = torchmetrics.MeanAbsoluteError()
        self.final_val_error_ratio = None
    
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        return self.model(x, edge_index, batch, edge_attr=edge_attr).view(-1)  # shape: (batch_size,)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def shared_step(self, batch, stage: str):
        """
        define the shared steps of train, validation, and test
        """
        preds = self.forward(batch)
        targets = batch.y
        
        # # unormalize 
        # if self.target_mean is not None and self.target_std is not None:
        #     preds = preds * self.target_std + self.target_mean
        #     targets = targets * self.target_std + self.target_mean
        # else:
        #     preds = preds
        #     targets = targets

        # define the metrics to track
        loss = self.loss_module(preds, targets)
        mae = self.mae_metric(preds, targets)
        error_ratio = mae / CHEMICAL_ACCURACY_MU

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_mae", mae, prog_bar=True)
        self.log(f"{stage}_error_ratio", error_ratio, prog_bar=True)

        if stage == 'val':
            self.final_val_error_ratio = error_ratio.detach().cpu().item()

        return loss
    

    def training_step(self, batch, batch_idx):
        # print(f"[Epoch {self.current_epoch}] Training batch {batch_idx}")
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, stage="test")


def train_graph_regressor(model_name, dataset, train_loader, val_loader, test_loader, **model_kwargs):
    pl.seed_everything(42)

    # Graph-level → batch size > 1
    # graph_loader = geom_data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Set checkpoint directory
    root_dir = os.path.join(model_path, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                monitor="val_mae",  # tracking mean absolute error
                mode="min"
            )
        ],
        accelerator=ACCELERATOR,
        devices=AVAIL_GPUS,
        max_epochs=30,
        enable_progress_bar=True,
    )

    # target_mean = dataset.data.y.mean()
    # target_std = dataset.data.y.std()

    model = GraphLevelGNN(
        model_name=model_name,
        **model_kwargs
    )


    # Split dataset manually if not already split
    if hasattr(dataset, "train_idx"):
        train_loader = geom_data.DataLoader(dataset[dataset.train_idx], batch_size=32, shuffle=True)
        val_loader = geom_data.DataLoader(dataset[dataset.val_idx], batch_size=32)
        test_loader = geom_data.DataLoader(dataset[dataset.test_idx], batch_size=32)
    else:
        # 80/10/10 split
        num_samples = 50000
        dataset = dataset[:num_samples]
        total = len(dataset)
        train_len = int(total * 0.8)
        val_len = int(total * 0.1)
        test_len = total - train_len - val_len
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])
        train_loader = geom_data.DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = geom_data.DataLoader(val_set, batch_size=32)
        test_loader = geom_data.DataLoader(test_set, batch_size=32)

    # Train
    trainer.fit(
        model,
        train_loader,
        val_loader,
        # ckpt_path=None
    )

    # Test using best checkpoint
    best_model_path = trainer.checkpoint_callback.best_model_path
    model = GraphLevelGNN.load_from_checkpoint(best_model_path)

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)[0]
    return model, {
        "test_loss": test_result["test_loss"],
        "test_mae": test_result["test_mae"],
        "test_error_ratio": test_result["test_error_ratio"]
    }

