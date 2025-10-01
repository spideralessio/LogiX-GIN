from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from tell import LogicalLayer

import torch

from torch import Tensor


def gumbel_sigmoid(logits, tau = 1, hard = False, threshold = 0.5, deterministic=False):
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    if deterministic: gumbels=0
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.15, nogumbel=False):
        super(GIN, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(num_features if i==0 else hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU() if nogumbel else torch.nn.Identity()
                ),
                init_eps=1
            )
            self.convs.append(conv)
        self.fc1 = torch.nn.Linear(num_layers*3*hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.nogumbel=nogumbel

    def forward(self, x, edge_index, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            if not self.nogumbel:
                x = gumbel_sigmoid(x, tau=tau, hard=True, deterministic=deterministic)
            xs.append(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def forward_e(self, x, edge_index, batch=None, tau=1, deterministic=False, *args, **kwargs):
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
        ret_x = []
        ret_y = []
        xs = []
        for conv in self.convs:
            ret_x.append(x)
            x = conv(x, edge_index)
            if not self.nogumbel:
                x = gumbel_sigmoid(x, tau=tau, hard=True)
            xs.append(x)
            ret_y.append(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        ret_x.append(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = torch.sigmoid(self.fc2(x))        
        ret_y.append(x)
        return ret_x, ret_y



class GINTELL(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=3, hidden_dim=64, dropout=0.1, dummy_phi_in=False):
        super(GINTELL, self).__init__()
        self.num_features, self.num_classes = num_features, num_classes
        self.convs = torch.nn.ModuleList()
        self.lls = []
        for i in range(num_layers):
            conv = GINConv(
                torch.nn.Sequential(
                    LogicalLayer(2*num_features if i==0 else 2*hidden_dim, hidden_dim, dummy_phi_in=(dummy_phi_in if i==0 else False)),
                ),
                init_eps=1,
                # aggr=['mean']
            )
            self.convs.append(conv)
        # self.input_bnorm = torch.nn.BatchNorm1d(num_features, affine=False)
        # self.output_bnorm = torch.nn.BatchNorm1d(num_layers*3*hidden_dim, affine=False)
        self.input_bnorm = torch.nn.Identity()
        self.output_bnorm = torch.nn.Identity()
        self.fc = LogicalLayer(2*num_layers*3*hidden_dim, num_classes, dummy_phi_in=False)
        # self.fc = torch.nn.Sequential(torch.nn.Linear(num_layers*3*hidden_dim, num_classes), torch.nn.Sigmoid())
    
        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        x = self.input_bnorm(x)
        xs = []
        for i, conv in enumerate(self.convs):
            if i!=0:self.dropout(x)
            x = conv(torch.hstack([x, 1-x]), edge_index)
            if discrete:
                indices = (x > 0.5).nonzero(as_tuple=True)
                x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format)
                x_hard[indices[0], indices[1]] = 1.0
                x = x_hard - x.detach() + x
            xs.append(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        x = self.output_bnorm(x)
        x = self.fc(torch.hstack([x, 1-x]))
        # x = self.fc(x)
        return x

    def forward_e(self, x, edge_index, batch, discrete=False, *args, **kwargs):
        x = self.input_bnorm(x)
        ret_x = []
        ret_y = []
        xs = []
        for i, conv in enumerate(self.convs):
            ret_x.append(x)
            if i!=0:self.dropout(x)
            x = conv(torch.hstack([x, 1-x]), edge_index)
            if discrete:
                indices = (x > 0.5).nonzero(as_tuple=True)
                x_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format)
                x_hard[indices[0], indices[1]] = 1.0
                x = x_hard - x.detach() + x
            xs.append(x)
            ret_y.append(x)

        x_mean = global_mean_pool(torch.hstack(xs), batch)
        x_max = global_max_pool(torch.hstack(xs), batch)
        x_sum = global_add_pool(torch.hstack(xs), batch)
        x = torch.hstack([x_mean, x_max, x_sum])
        x = self.output_bnorm(x)
        ret_x.append(x)
        x = self.fc(torch.hstack([x, 1-x]), discrete_output=False)
        # x = self.fc(x)
        ret_y.append(x)
        return ret_x, ret_y
