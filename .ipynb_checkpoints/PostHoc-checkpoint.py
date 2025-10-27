#!/usr/bin/env python
# coding: utf-8

from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from syn_dataset import SynGraphDataset
from spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import argparse
import pickle
import json
import io
from model import GIN
from train_baseline import test_epoch
from tell_sigmoid import LogicalLayer
import sys
import pickle
import torch
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


dataset_name = sys.argv[1]
seed = int(sys.argv[2])
def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=True')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname


results_path = os.path.join(get_best_baseline_path(dataset_name), str(seed))
data = pickle.load(open(os.path.join(results_path, 'data.pkl'), 'rb'))
args = json.load(open(os.path.join(results_path, 'args.json'), 'r'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = get_dataset(dataset_name)
num_classes = dataset.num_classes
num_features = dataset.num_features
num_layers = args['num_layers']
hidden_dim = args['hidden_dim']
model = GIN(num_classes=num_classes, num_features=num_features, num_layers=num_layers, hidden_dim=hidden_dim, nogumbel=True, dropout=0.1)
model.load_state_dict(torch.load(os.path.join(results_path, 'best.pt'), map_location=device))
model = model.to(device)
train_indices = data['train_indices']
val_indices = data['val_indices']
test_indices = data['test_indices']
train_dataset = dataset[train_indices]
val_dataset = dataset[val_indices]
test_dataset = dataset[test_indices]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_acc = test_epoch(model, val_loader, device)
test_acc = test_epoch(model, test_loader, device)


def inverse_sigmoid(x):
    """Computes the inverse of the sigmoid function (logit function)."""
    return torch.log(x / (1 - x))
#!/usr/bin/env python
# coding: utf-8

from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from syn_dataset import SynGraphDataset
from spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import argparse
import pickle
import json
import io
from model import GIN
from train_baseline import test_epoch
from tell_sigmoid import LogicalLayer
import sys
import pickle
import torch
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


dataset_name = sys.argv[1]
seed = int(sys.argv[2])
def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=True')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname


results_path = os.path.join(get_best_baseline_path(dataset_name), str(seed))
data = pickle.load(open(os.path.join(results_path, 'data.pkl'), 'rb'))
args = json.load(open(os.path.join(results_path, 'args.json'), 'r'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = get_dataset(dataset_name)
num_classes = dataset.num_classes
num_features = dataset.num_features
num_layers = args['num_layers']
hidden_dim = args['hidden_dim']
model = GIN(num_classes=num_classes, num_features=num_features, num_layers=num_layers, hidden_dim=hidden_dim, nogumbel=True, dropout=0.1)
model.load_state_dict(torch.load(os.path.join(results_path, 'best.pt'), map_location=device))
model = model.to(device)
train_indices = data['train_indices']
val_indices = data['val_indices']
test_indices = data['test_indices']
train_dataset = dataset[train_indices]
val_dataset = dataset[val_indices]
test_dataset = dataset[test_indices]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_acc = test_epoch(model, val_loader, device)
test_acc = test_epoch(model, test_loader, device)


def inverse_sigmoid(x):
    """Computes the inverse of the sigmoid function (logit function)."""
    return torch.log(x / (1 - x))

torch.no_grad()
def find_logic_rules(w, t_in, t_out, activations=None, max_rule_len=10, max_rules=100, min_support=1):
    w = w.clone()
    t_in = t_in.clone()
    t_out = t_out.clone()
    t_out = t_out.item()
    ordering_scores = w
    sorted_idxs = torch.argsort(ordering_scores, 0, descending=True)
    mask = w > 1e-5
    if activations is not None:
        mask = mask & (activations.sum(0) >= min_support)
    total_result = set()

    # Filter and sort indices based on the mask
    idxs_to_visit = sorted_idxs[mask[sorted_idxs]]
    if idxs_to_visit.numel() == 0:
        return total_result

    # Sort weights based on the filtered indices
    sorted_weights = w[idxs_to_visit]
    current_combination = []
    result = set()

    def find_logic_rules_recursive(index, current_sum):
        # Stop if the maximum number of rules has been reached
        if len(result) >= max_rules:
            return

        if len(current_combination) > max_rule_len:
            return

        # Check if the current combination satisfies the condition
        if current_sum >= t_out:
            c = idxs_to_visit[current_combination].cpu().detach().tolist()
            c = tuple(sorted(c))
            result.add(c)
            return

        # Prune if remaining weights can't satisfy t_out
        remaining_max_sum = current_sum + sorted_weights[index:].sum()
        if remaining_max_sum < t_out:
            return

        # Explore further combinations
        for i in range(index, idxs_to_visit.shape[0]):
            # Prune based on activations if provided
            if activations is not None and len(current_combination) > 0 and activations[:, idxs_to_visit[current_combination + [i]]].all(-1).sum().item() < min_support:
                continue

            current_combination.append(i)
            find_logic_rules_recursive(i + 1, current_sum + sorted_weights[i])
            current_combination.pop()

    # Start the recursive process
    find_logic_rules_recursive(0, 0)
    return result


def extract_rules(self, feature=None, activations=None, max_rule_len=float('inf'), max_rules=100, min_support=1, out_threshold=0.5):
    ws = self.weight
    t_in = self.phi_in.t
    t_out = -self.b + inverse_sigmoid(torch.tensor(out_threshold))

    rules = []
    if feature is None:
        features = range(self.out_features)
    else:
        features = [feature]
    for i in features:
        w = ws[i].to('cpu')
        ti = t_in.to('cpu')
        to = t_out[i].to('cpu')
        rules.append(find_logic_rules(w, ti, to, activations, max_rule_len, max_rules, min_support))

    return rules



def get_blues_color(value):
    cmap = plt.get_cmap("Blues")  # Get the Blues colormap
    return cmap(value)  # Return the RGBA color

def plot_activations(batch_ids, batch, attr, soft=False):
    if type(batch_ids) != list:
        batch_ids = [batch_ids]
    if soft: attr = (attr-attr.min() + 1e-6)/(attr.max()-attr.min()+1e-6)
    fig, axs = plt.subplots(1, len(batch_ids), figsize=(16*len(batch_ids), 8))
    if type(axs) != np.ndarray: axs = np.array([axs])
    for i, batch_id in enumerate(batch_ids):
        node_mask = batch.batch == batch_id  # Get nodes where batch == 0
        node_indices = torch.nonzero(node_mask, as_tuple=True)[0]
        
        subgraph_edge_mask = (batch.batch[batch.edge_index[0]] == batch_id) & \
                             (batch.batch[batch.edge_index[1]] == batch_id)
        subgraph_edges = batch.edge_index[:, subgraph_edge_mask]
        
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        remapped_edges = torch.tensor([[node_mapping[e.item()] for e in edge] for edge in subgraph_edges.T])
        
        G = nx.Graph()
        G.add_edges_from(remapped_edges.numpy())
        
        nx.set_node_attributes(G, {v: k for k, v in node_mapping.items()}, "original_id")
        
        node_colors = []
        node_borders = []
        
        for node in G.nodes:
            node_colors.append(get_blues_color(attr[batch.batch==batch_id][node]))  # Fill color
            
        
        pos = nx.kamada_kawai_layout(G) 
        
        nx.draw(
            G, pos,
            node_color=node_colors,
            edgecolors=node_borders,  # Border colors
            node_size=700,
            with_labels=False,
            ax = axs[i]
        )
        
        axs[i].set_title(f"Class = {batch.y[batch_id]}")
    plt.show()



import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

def calculate_fidelity(data, node_mask, model, remove_nodes=True, top_k=None):

    device = next(model.parameters()).device
    data = data.to(device)

    # Compute sparsity
    total_nodes = data.x.shape[0]
    sparsity = 1 - node_mask.sum().item() / total_nodes if total_nodes > 0 else 0

    # Get original predictions
    original_pred = model(data.x.float(), data.edge_index)
    # original_pred = F.softmax(original_pred, dim=1)
    label = original_pred.argmax(-1).item()
    
    if remove_nodes:
        masked_edge_index, _ = subgraph(node_mask == 0, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
        masked_pred = model(data.x[node_mask == 0], masked_edge_index)
    else:
        masked_x = data.x.clone()
        masked_x[node_mask==1] = 0
        masked_pred = model(masked_x, data.edge_index)

    if remove_nodes:
        masked_edge_index, _ = subgraph(node_mask == 1, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
        retained_pred = model(data.x[node_mask == 1], masked_edge_index)
    else:
        masked_x = data.x.clone()
        masked_x[node_mask==0] = 0
        retained_pred = model(masked_x, data.edge_index)
    # retained_pred = F.softmax(retained_pred, dim=1)


    # Compute Fidelity+ and Fidelity-
    # inv_fidelity = (original_pred[:, label] - 
    #                              retained_pred[:, label]).mean().item()

    # fidelity = (original_pred[:, label] - 
    #                              masked_pred[:, label]).mean().item()

    inv_fidelity = (original_pred.argmax(-1) != 
                                 retained_pred.argmax(-1)).float().item()

    fidelity = (original_pred.argmax(-1)  != 
                                 masked_pred.argmax(-1) ).float().item()

    n_fidelity = inv_fidelity*sparsity
    n_inv_fidelity = inv_fidelity*(1-sparsity)
    
    # Compute HFidelity (harmonic mean of Fidelity+ and Fidelity-)
    hfidelity = ((1+n_fidelity) * (1-n_inv_fidelity)) / (2 + n_fidelity - n_inv_fidelity) if (1 + n_fidelity - n_inv_fidelity) != 0 else 0

    return {
        "Fidelity": fidelity,
        "InvFidelity": inv_fidelity,
        "HFidelity": hfidelity
    }


def calculate_fidelity_topk(data, node_soft_mask, model, top_k, remove_nodes=True):

    device = next(model.parameters()).device
    data = data.to(device)

    # Compute sparsity
    total_nodes = data.x.shape[0]

    # Get original predictions
    original_pred = model(data.x.float(), data.edge_index)
    # original_pred = F.softmax(original_pred, dim=1)
    label = original_pred.argmax(-1).item()

    

    node_mask = torch.zeros_like(node_soft_mask)
    node_mask[torch.topk(node_soft_mask, top_k).indices] = 1
    
    if remove_nodes:
        masked_edge_index, _ = subgraph(node_mask == 0, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
        masked_pred = model(data.x[node_mask == 0], masked_edge_index)
    else:
        masked_x = data.x.clone()
        masked_x[node_mask==1] = 0
        masked_pred = model(masked_x, data.edge_index)

    node_mask = torch.zeros_like(node_soft_mask)
    node_mask[torch.topk(node_soft_mask, total_nodes-top_k).indices] = 1
    
    if remove_nodes:
        masked_edge_index, _ = subgraph(node_mask == 1, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
        retained_pred = model(data.x[node_mask == 1], masked_edge_index)
    else:
        masked_x = data.x.clone()
        masked_x[node_mask==0] = 0
        retained_pred = model(masked_x, data.edge_index)


    # inv_fidelity = (original_pred[:, label] - 
    #                              retained_pred[:, label]).mean().item()

    # fidelity = (original_pred[:, label] - 
    #                              masked_pred[:, label]).mean().item()

    inv_fidelity = (original_pred.argmax(-1) != 
                                 retained_pred.argmax(-1)).float().item()

    fidelity = (original_pred.argmax(-1)  != 
                                 masked_pred.argmax(-1) ).float().item()
    return {
        "Fidelity": fidelity,
        "InvFidelity": inv_fidelity,
    }




# In[105]:


def node_imp_from_edge_imp(edge_index, n_nodes, edge_imp):
    node_imp = torch.zeros(n_nodes)
    for i in range(n_nodes):
        node_imp[i] = edge_imp[(edge_index[0]==i) | (edge_index[1]==i)].mean()
    return node_imp
        
# def generate_hard_masks(soft_mask):
#     sparsity_levels = torch.arange(0.5,1, 0.05)
#     hard_masks = []
#     for sparsity in sparsity_levels:
#         threshold = np.percentile(soft_mask, sparsity * 100)
#         hard_mask = (soft_mask > threshold).int()
#         if hard_mask.sum() == 0:
#             hard_mask = (soft_mask > soft_mask.min()).int()
#         hard_masks.append(hard_mask)
#     return list(zip(sparsity_levels, hard_masks))

import torch
import numpy as np

import torch

def generate_hard_masks(soft_mask):
    soft_mask_flat = soft_mask.flatten()
    total_elements = soft_mask_flat.numel()
    sparsity_levels = torch.arange(0.5, 1.0, 0.05)
    hard_masks = []

    # Get sorted indices (ascending: lowest values first)
    sorted_indices = torch.argsort(soft_mask_flat)

    for sparsity in sparsity_levels:
        num_to_mask = int(sparsity.item() * total_elements)
        mask_flat = torch.ones_like(soft_mask_flat, dtype=torch.int)
        
        if num_to_mask >= total_elements:
            num_to_mask = total_elements - 1  # keep at least one element
        
        # Zero out the lowest `num_to_mask` elements
        mask_flat[sorted_indices[:num_to_mask]] = 0
        
        # Reshape to original shape
        hard_mask = mask_flat.view_as(soft_mask)
        hard_masks.append(hard_mask)

    return list(zip(sparsity_levels, hard_masks))

os.makedirs(f'post_hoc/{dataset_name}/{seed}', exist_ok=True)


import numpy as np
import torch
import torch.nn as nn
# import tqdm
import time

from typing import Optional
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv

from graphxai.explainers._base import _BaseExplainer
from graphxai.utils import Explanation, node_mask_from_edge_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

class PGExplainer(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG
    """
    def __init__(self, model: nn.Module, emb_layer_name: str = None,
                 explain_graph: bool = False,
                 coeff_size: float = 0.01, coeff_ent: float = 5e-4,
                 t0: float = 5.0, t1: float = 2.0,
                 lr: float = 0.003, max_epochs: int = 20, eps: float = 1e-3,
                 num_hops: int = None, in_channels = None):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
                The output of the model should be unnormalized class score.
                For example, last layer = CNConv or Linear.
            emb_layer_name (str, optional): name of the embedding layer
                If not specified, use the last but one layer by default.
            explain_graph (bool): whether the explanation is graph-level or node-level
            coeff_size (float): size regularization to constrain the explanation size
            coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            t0 (float): the temperature at the first epoch
            t1 (float): the temperature at the final epoch
            lr (float): learning rate to train the explanation network
            max_epochs (int): number of epochs to train the explanation network
            num_hops (int): number of hops to consider for node-level explanation
        """
        super().__init__(model, emb_layer_name)

        # Parameters for PGExplainer
        self.explain_graph = explain_graph
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
        self.lr = lr
        self.eps = eps
        self.max_epochs = max_epochs
        self.num_hops = self.L if num_hops is None else num_hops

        # Explanation model in PGExplainer

        mult = 2 # if self.explain_graph else 3

        # if in_channels is None:
        #     if isinstance(self.emb_layer, GCNConv):
        #         in_channels = mult * self.emb_layer.out_channels
        #     elif isinstance(self.emb_layer, GINConv):
        #         in_channels = mult * self.emb_layer.nn.out_features
        #     elif isinstance(self.emb_layer, torch.nn.Linear):
        #         in_channels = mult * self.emb_layer.out_features
        #     else:
        #         fmt_string = 'PGExplainer not implemented for embedding layer of type {}, please provide in_channels directly.'
        #         raise NotImplementedError(fmt_string.format(type(self.emb_layer)))
        in_channels = mult*hidden_dim*num_layers
        self.elayers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.ReLU()),
             nn.Linear(64, 1)]).to(device)

    def __concrete_sample(self, log_alpha: torch.Tensor,
                          beta: float = 1.0, training: bool = True):
        """
        Sample from the instantiation of concrete distribution when training.

        Returns:
            training == True: sigmoid((log_alpha + noise) / beta)
            training == False: sigmoid(log_alpha)
        """
        if training:
            random_noise = torch.rand(log_alpha.shape).to(device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def __emb_to_edge_mask(self, emb: torch.Tensor,
                           x: torch.Tensor, edge_index: torch.Tensor,
                           node_idx: int = None,
                           forward_kwargs: dict = {},
                           tmp: float = 1.0, training: bool = False):
        """
        Compute the edge mask based on embedding.

        Returns:
            prob_with_mask (torch.Tensor): the predicted probability with edge_mask
            edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
        """
#        if not self.explain_graph and node_idx is None:
#            raise ValueError('node_idx should be provided.')

        with torch.set_grad_enabled(training):
            # Concat relevant node embeddings
            # import ipdb; ipdb.set_trace()
            U, V = edge_index  # edge (u, v), U = (u), V = (v)
            h1 = emb[U]
            h2 = emb[V]
            if self.explain_graph:
                h = torch.cat([h1, h2], dim=1)
            else:
                h3 = emb.repeat(h1.shape[0], 1)
                h = torch.cat([h1, h2], dim=1)

            # Calculate the edge weights and set the edge mask
            for elayer in self.elayers:
                h = elayer.to(device)(h)
            h = h.squeeze()
            edge_weights = self.__concrete_sample(h, tmp, training)
            n = emb.shape[0]  # number of nodes
            mask_sparse = torch.sparse_coo_tensor(
                edge_index, edge_weights, (n, n))
            # Not scalable
            self.mask_sigmoid = mask_sparse.to_dense()
            # Undirected graph
            sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
            edge_mask = sym_mask[edge_index[0], edge_index[1]]
            #print('edge_mask', edge_mask)
            self._set_masks(x, edge_index, edge_mask)

        # Compute the model prediction with edge mask
        # with torch.no_grad():
        #     tester = self.model(x, edge_index)
        #     print(tester)
        prob_with_mask = self._predict(x, edge_index,
                                       forward_kwargs=forward_kwargs,
                                       return_type='prob')
        self._clear_masks()

        return prob_with_mask, edge_mask


    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        y = y[subset]
        return x, edge_index, y, subset, kwargs


    def train_explanation_model(self, dataset: Data, forward_kwargs: dict = {}):
        """
        Train the explanation model.
        """
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)

        def loss_fn(prob: torch.Tensor, ori_pred: int):
            # Maximize the probability of predicting the label (cross entropy)
            loss = -torch.log(prob[ori_pred] + 1e-6)
            # Size regularization
            edge_mask = self.mask_sigmoid
            loss += self.coeff_size * torch.sum(edge_mask)
            # Element-wise entropy regularization
            # Low entropy implies the mask is close to binary
            edge_mask = edge_mask * 0.99 + 0.005
            entropy = - edge_mask * torch.log(edge_mask) \
                - (1 - edge_mask) * torch.log(1 - edge_mask)
            loss += self.coeff_ent * torch.mean(entropy)

            return loss

        if self.explain_graph:  # Explain graph-level predictions of multiple graphs
            # Get the embeddings and predicted labels
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(device)
                    pred_label = self._predict(data.x.float(), data.edge_index,
                                               forward_kwargs=forward_kwargs)
                    # emb = self._get_embedding(data.x.float(), data.edge_index,
                    #                           forward_kwargs=forward_kwargs)

                    xs, ys = self.model.forward_e(data.x.float(), data.edge_index)
                    emb = torch.hstack(ys[:-1])
                    # OWEN inserting:
                    emb_dict[gid] = emb.to(device) # Add embedding to embedding dictionary
                    ori_pred_dict[gid] = pred_label

            # Train the mask generator
            duration = 0.0
            last_loss = 0.0
            for epoch in range(self.max_epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(device)
                    prob_with_mask, _ = self.__emb_to_edge_mask(
                        emb_dict[gid], data.x.float(), data.edge_index,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, training=True)
                    loss_tmp = loss_fn(prob_with_mask.squeeze(), ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob_with_mask.argmax(-1).item()
                    pred_list.append(pred_label)
                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')
                if abs(loss - last_loss) < self.eps:
                    break
                last_loss = loss

        else:  # Explain node-level predictions of a graph
            data = dataset.to(device)
            X = data.x
            EIDX = data.edge_index

            # Get the predicted labels for training nodes
            with torch.no_grad():
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                # pred_dict = {}
                label = self._predict(X, EIDX, forward_kwargs=forward_kwargs)
                pred_dict = dict(zip(explain_node_index_list, label[explain_node_index_list]))
                # for node_idx in tqdm.tqdm(explain_node_index_list):
                #     pred_dict[node_idx] = label[node_idx]

            # Train the mask generator
            duration = 0.0
            last_loss = 0.0
            x_dict = {}
            edge_index_dict = {}
            node_idx_dict = {}
            emb_dict = {}
            for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, self.num_hops, EIDX, relabel_nodes=True, num_nodes=data.x.shape[0])
#                 new_node_index.append(int(torch.where(subset == node_idx)[0]))
                x_dict[node_idx] = X[subset].to(device)
                edge_index_dict[node_idx] = sub_edge_index.to(device)
                emb = self._get_embedding(X[subset], sub_edge_index,forward_kwargs=forward_kwargs)
                emb_dict[node_idx] = emb.to(device)
                node_idx_dict[node_idx] = int(torch.where(subset==node_idx)[0])

            for epoch in range(self.max_epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                self.elayers.train()
                tic = time.perf_counter()

                for iter_idx, node_idx in tqdm.tqdm(enumerate(x_dict.keys())):
                    prob_with_mask, _ = self.__emb_to_edge_mask(
                        emb_dict[node_idx], 
                        x = x_dict[node_idx], 
                        edge_index = edge_index_dict[node_idx], 
                        node_idx = node_idx,
                        forward_kwargs=forward_kwargs,
                        tmp=tmp, 
                        training=True)
                    loss_tmp = loss_fn(prob_with_mask[node_idx_dict[node_idx]], pred_dict[node_idx])
                    loss_tmp.backward()
                    # loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
#                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
                # if abs(loss - last_loss) < self.eps:
                #     break
                # last_loss = loss

            print(f"training time is {duration:.5}s")

    def get_explanation_node(self, node_idx: int, x: torch.Tensor,
                             edge_index: torch.Tensor, label: torch.Tensor = None,
                             y = None,
                             forward_kwargs: dict = {}, **_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): index of the node to be explained
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [n]): k-hop node importance
            khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        if self.explain_graph:
            raise Exception('For graph-level explanations use `get_explanation_graph`.')
        label = self._predict(x, edge_index) if label is None else label

        khop_info = _, _, _, sub_edge_mask = \
            k_hop_subgraph(node_idx, self.num_hops, edge_index,
                           relabel_nodes=False, num_nodes=x.shape[0])
        emb = self._get_embedding(x, edge_index, forward_kwargs=forward_kwargs)
        _, edge_mask = self.__emb_to_edge_mask(
            emb, x, edge_index, node_idx, forward_kwargs=forward_kwargs,
            tmp=2, training=False)
        edge_imp = edge_mask[sub_edge_mask]

        exp = Explanation(
            node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1], edge_imp.bool()),
            edge_imp = edge_imp,
            node_idx = node_idx
        )

        exp.set_enclosing_subgraph(khop_info)

        return exp

    def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
                              label: Optional[torch.Tensor] = None,
                              y = None,
                              forward_kwargs: dict = {},
                              top_k: int = 10):
        """
        Explain a whole-graph prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            label (torch.Tensor, optional, [n x ...]): labels to explain
                If not provided, we use the output of the model.
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index
            top_k (int): number of edges to include in the edge-importance explanation

        Returns:
            exp (dict):
                exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
                exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
                exp['node_imp'] (torch.Tensor, [m]): k-hop node importance

        """
        if not self.explain_graph:
            raise Exception('For node-level explanations use `get_explanation_node`.')

        label = self._predict(x, edge_index,
                              forward_kwargs=forward_kwargs) if label is None else label

        with torch.no_grad():
            # emb = self._get_embedding(x, edge_index,
            #                           forward_kwargs=forward_kwargs)
            xs, ys = self.model.forward_e(data.x.float(), data.edge_index)
            emb = torch.hstack(ys[:-1])
            _, edge_mask = self.__emb_to_edge_mask(
                emb, x, edge_index, forward_kwargs=forward_kwargs,
                tmp=1.0, training=False)

        #exp['edge_imp'] = edge_mask

        exp = Explanation(
            node_imp = node_mask_from_edge_mask(torch.arange(x.shape[0], device=x.device), edge_index),
            edge_imp = edge_mask
        )

        exp.set_whole_graph(Data(x=x, edge_index=edge_index))

        return exp

    def get_explanation_link(self):
        """
        Explain a link prediction.
        """
        raise NotImplementedError()


from concurrent.futures import ThreadPoolExecutor

def explain_pg_explainer(model, data):
    r = {
        'data': data,
        'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
        'res' : {}
    }
    data = data.to(device)
    model=model.to(device)
    pgexplainer = PGExplainer(model, explain_graph=True)
    exp = pgexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index)
    soft_mask = node_imp_from_edge_imp(data.edge_index, data.x.shape[0], exp.edge_imp)
    soft_mask[soft_mask!=soft_mask]=0
    r['soft_mask'] = soft_mask
    hard_masks = generate_hard_masks(soft_mask)
    for sparsity, hard_mask in hard_masks:
        sparsity=sparsity.item()
        r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
        r['res'][sparsity]['hard_mask'] = hard_mask

    r['res_topk'] = {
        1: calculate_fidelity_topk(data, soft_mask, model,1),
        3: calculate_fidelity_topk(data, soft_mask, model,3),
        5: calculate_fidelity_topk(data, soft_mask, model,5)
    }
    return r
    
pg_explainer_results = []
for data in tqdm(test_dataset):
    try:
        data = data.to(device)
        data.x = data.x.float()
        r = explain_pg_explainer(model, data)
        pg_explainer_results.append(r)
    except: pass


with open(f'post_hoc/{dataset_name}/{seed}/pgexplainer.pkl', 'wb') as f:
    pickle.dump(pg_explainer_results, f)
pg_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/pgexplainer.pkl', 'rb'))






from graphxai.explainers import GNNExplainer

def explain_gnn_explainer(model, data):
    r = {
        'data': data,
        'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
        'res' : {}
    }
    data = data.to(device)
    model=model.to(device)
    gnnexplainer = GNNExplainer(model)
    exp = gnnexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index)
    soft_mask = node_imp_from_edge_imp(data.edge_index, data.x.shape[0], exp.edge_imp)
    soft_mask[soft_mask!=soft_mask]=0
    r['soft_mask'] = soft_mask
    hard_masks = generate_hard_masks(soft_mask)
    for sparsity, hard_mask in hard_masks:
        sparsity=sparsity.item()
        r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
        r['res'][sparsity]['hard_mask'] = hard_mask

    r['res_topk'] = {
        1: calculate_fidelity_topk(data, soft_mask, model,1),
        3: calculate_fidelity_topk(data, soft_mask, model,3),
        5: calculate_fidelity_topk(data, soft_mask, model,5)
    }
    return r
    
gnn_explainer_results = []
for data in tqdm(test_dataset):
    data = data.to(device)
    data.x = data.x.float()
    try:
        r = explain_gnn_explainer(model, data)
        gnn_explainer_results.append(r)
    except: pass

with open(f'post_hoc/{dataset_name}/{seed}/gnnexplainer.pkl', 'wb') as f:
    pickle.dump(gnn_explainer_results, f)
gnn_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/gnnexplainer.pkl', 'rb'))



from graphxai.explainers import IntegratedGradExplainer

def explain_ig_explainer(model, data):
    r = {
        'data': data,
        'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
        'res' : {}
    }
    data = data.to(device)
    model=model.to(device)
    igexplainer = IntegratedGradExplainer(model, torch.nn.CrossEntropyLoss())
    exp = igexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index, label=torch.tensor(r['pred'].argmax(-1)).to(device))
    soft_mask = exp.node_imp.detach().cpu()
    r['soft_mask'] = soft_mask
    hard_masks = generate_hard_masks(soft_mask)
    for sparsity, hard_mask in hard_masks:
        sparsity=sparsity.item()
        r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
        r['res'][sparsity]['hard_mask'] = hard_mask
    r['res_topk'] = {
        1: calculate_fidelity_topk(data, soft_mask, model,1),
        3: calculate_fidelity_topk(data, soft_mask, model,3),
        5: calculate_fidelity_topk(data, soft_mask, model,5)
    }
    return r
    
ig_explainer_results = []
for data in tqdm(test_dataset):
    data = data.to(device)
    data.x = data.x.float()
    try:
        r = explain_ig_explainer(model, data)
        ig_explainer_results.append(r)
    except: pass


# In[27]:


with open(f'post_hoc/{dataset_name}/{seed}/ig.pkl', 'wb') as f:
    pickle.dump(ig_explainer_results, f)
ig_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/ig.pkl', 'rb'))


# In[28]:


from graphxai.explainers import PGExplainer


# # GStarX

# In[30]:


import torch
from torch_geometric.utils import subgraph, to_dense_adj
import os
import pytz
import logging
import numpy as np
import random
import torch
import networkx as nx
import copy
from rdkit import Chem
from datetime import datetime
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.data import Data, Batch, Dataset, DataLoader

# For associated game
from itertools import combinations
from scipy.sparse.csgraph import connected_components as cc

def set_partitions(iterable, k=None, min_size=None, max_size=None):
    """
    Yield the set partitions of *iterable* into *k* parts. Set partitions are
    not order-preserving.

    >>> iterable = 'abc'
    >>> for part in set_partitions(iterable, 2):
    ...     print([''.join(p) for p in part])
    ['a', 'bc']
    ['ab', 'c']
    ['b', 'ac']


    If *k* is not given, every set partition is generated.

    >>> iterable = 'abc'
    >>> for part in set_partitions(iterable):
    ...     print([''.join(p) for p in part])
    ['abc']
    ['a', 'bc']
    ['ab', 'c']
    ['b', 'ac']
    ['a', 'b', 'c']

    if *min_size* and/or *max_size* are given, the minimum and/or maximum size
    per block in partition is set.

    >>> iterable = 'abc'
    >>> for part in set_partitions(iterable, min_size=2):
    ...     print([''.join(p) for p in part])
    ['abc']
    >>> for part in set_partitions(iterable, max_size=2):
    ...     print([''.join(p) for p in part])
    ['a', 'bc']
    ['ab', 'c']
    ['b', 'ac']
    ['a', 'b', 'c']

    """
    L = list(iterable)
    n = len(L)
    if k is not None:
        if k < 1:
            raise ValueError(
                "Can't partition in a negative or zero number of groups"
            )
        elif k > n:
            return

    min_size = min_size if min_size is not None else 0
    max_size = max_size if max_size is not None else n
    if min_size > max_size:
        return

    def set_partitions_helper(L, k):
        n = len(L)
        if k == 1:
            yield [L]
        elif n == k:
            yield [[s] for s in L]
        else:
            e, *M = L
            for p in set_partitions_helper(M, k - 1):
                yield [[e], *p]
            for p in set_partitions_helper(M, k):
                for i in range(len(p)):
                    yield p[:i] + [[e] + p[i]] + p[i + 1 :]

    if k is None:
        for k in range(1, n + 1):
            yield from filter(
                lambda z: all(min_size <= len(bk) <= max_size for bk in z),
                set_partitions_helper(L, k),
            )
    else:
        yield from filter(
            lambda z: all(min_size <= len(bk) <= max_size for bk in z),
            set_partitions_helper(L, k),
        )


# For visualization
from typing import Union, List
from textwrap import wrap
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()


def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
    check_dir(log_path)

    tz = pytz.timezone("US/Pacific")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def get_graph_build_func(build_method):
    if build_method.lower() == "zero_filling":
        return graph_build_zero_filling
    elif build_method.lower() == "split":
        return graph_build_split
    elif build_method.lower() == "remove":
        return graph_build_remove
    else:
        raise NotImplementedError


"""
Graph building/Perturbation
`graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
"""


def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through masking the unselected nodes with zero features"""
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index


def graph_build_split(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through spliting the selected nodes from the original graph"""
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index


def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
    """subgraph building through removing the unselected nodes from the original graph"""
    ret_X = X[node_mask == 1]
    ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
    return ret_X, ret_edge_index


"""
Associated game of the HN value
Implementated using sparse tensor
"""


def get_ordered_coalitions(n):
    coalitions = sum(
        [[set(c) for c in combinations(range(n), k)] for k in range(1, n + 1)], []
    )
    return coalitions


def get_associated_game_matrix_M(coalitions, n, tau):
    indices = []
    values = []
    for i, s in enumerate(coalitions):
        for j, t in enumerate(coalitions):
            if i == j:
                indices += [[i, j]]
                values += [1 - (n - len(s)) * tau]
            elif len(s) + 1 == len(t) and s.issubset(t):
                indices += [[i, j]]
                values += [tau]
            elif len(t) == 1 and not t.issubset(s):
                indices += [[i, j]]
                values += [-tau]

    indices = torch.Tensor(indices).t()
    size = (2**n - 1, 2**n - 1)
    M = torch.sparse_coo_tensor(indices, values, size)
    return M


def get_associated_game_matrix_P(coalitions, n, adj):
    indices = []
    for i, s in enumerate(coalitions):
        idx_s = torch.LongTensor(list(s))
        num_cc, labels = cc(adj[idx_s, :][:, idx_s])
        cc_s = []
        for k in range(num_cc):
            cc_idx_s = (labels == k).nonzero()[0]
            cc_s += [set((idx_s[cc_idx_s]).tolist())]
        for j, t in enumerate(coalitions):
            if t in cc_s:
                indices += [[i, j]]

    indices = torch.Tensor(indices).t()
    values = [1.0] * indices.shape[-1]
    size = (2**n - 1, 2**n - 1)

    P = torch.sparse_coo_tensor(indices, values, size)
    return P


def get_limit_game_matrix(H, exp_power=7, tol=1e-3, is_sparse=True):
    """
    Speed up the power computation by
    1. Use sparse matrices
    2. Put all tensors on cuda
    3. Compute powers exponentially rather than linearly
        i.e. H -> H^2 -> H^4 -> H^8 -> H^16 -> ...
    """
    i = 0
    diff_norm = tol + 1
    while i < exp_power and diff_norm > tol:
        if is_sparse:
            H_tilde = torch.sparse.mm(H, H)
        else:
            H_tilde = torch.mm(H, H)
        diff_norm = (H_tilde - H).norm()
        H = H_tilde
        i += 1
    return H_tilde


"""
khop or random sampling to generate subgraphs
"""


def sample_subgraph(
    data, max_sample_size, sample_method, target_node=None, k=0, adj=None
):
    if sample_method == "khop":
        # pick nodes within k-hops of target node. Hop by hop until reach max_sample_size
        if adj is None:
            adj = (
                to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                .detach()
                .cpu()
            )

        adj_self_loop = adj + torch.eye(data.num_nodes)
        k_hop_adj = adj_self_loop
        sampled_nodes = set()
        m = max_sample_size
        l = 0
        while k > 0 and l < m:
            k_hop_nodes = k_hop_adj[target_node].nonzero().view(-1).tolist()
            next_hop_nodes = list(set(k_hop_nodes) - sampled_nodes)
            sampled_nodes.update(next_hop_nodes[: m - l])
            l = len(sampled_nodes)
            k -= 1
            k_hop_adj = torch.mm(k_hop_adj, adj_self_loop)
        sampled_nodes = torch.tensor(list(sampled_nodes))

    elif sample_method == "random":  # randomly pick #max_sample_size nodes
        sampled_nodes = torch.randperm(data.num_nodes)[:max_sample_size]
    else:
        ValueError("Unknown sample method")

    sampled_x = data.x[sampled_nodes]
    sampled_edge_index, _ = subgraph(
        sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )
    sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index)
    sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]

    return sampled_nodes, sampled_data, sampled_adj


"""
Payoff computation
"""


def get_char_func(model, target_class, payoff_type="norm_prob", payoff_avg=None):
    def char_func(data):
        with torch.no_grad():
            logits = model(data.x.float(), data.edge_index)
            if payoff_type == "raw":
                payoff = logits[:, target_class]
            elif payoff_type == "prob":
                payoff = logits.softmax(dim=-1)[:, target_class]
            elif payoff_type == "norm_prob":
                prob = logits.softmax(dim=-1)[:, target_class]
                payoff = prob - payoff_avg[target_class]
            elif payoff_type == "log_prob":
                payoff = logits.log_softmax(dim=-1)[:, target_class]
            else:
                raise ValueError("unknown payoff type")
        return payoff

    return char_func


class MaskedDataset(Dataset):
    def __init__(self, data, mask, subgraph_building_func):
        super().__init__()

        self.num_nodes = data.num_nodes
        self.x = data.x
        self.edge_index = data.edge_index
        self.device = data.x.device
        self.y = data.y

        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)

        self.mask = mask.type(torch.float32).to(self.device)
        self.subgraph_building_func = subgraph_building_func

    def __len__(self):
        return self.mask.shape[0]

    def __getitem__(self, idx):
        masked_x, masked_edge_index = self.subgraph_building_func(
            self.x, self.edge_index, self.mask[idx]
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index)
        return masked_data


def get_coalition_payoffs(data, coalitions, char_func, subgraph_building_func):
    n = data.num_nodes
    masks = []
    for coalition in coalitions:
        mask = torch.zeros(n)
        mask[list(coalition)] = 1.0
        masks += [mask]

    coalition_mask = torch.stack(masks, axis=0)
    masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
    masked_dataloader = DataLoader(
        masked_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    masked_payoff_list = []
    for masked_data in masked_dataloader:
        masked_payoff_list.append(char_func(masked_data))

    masked_payoffs = torch.cat(masked_payoff_list, dim=0)
    return masked_payoffs


"""
Superadditive extension
"""


class TrieNode:
    def __init__(self, player, payoff=0, children=[]):
        self.player = player
        self.payoff = payoff
        self.children = children


class CoalitionTrie:
    def __init__(self, coalitions, n, v):
        self.n = n
        self.root = self.get_node(None, 0)
        for i, c in enumerate(coalitions):
            self.insert(c, v[i])

    def get_node(self, player, payoff):
        return TrieNode(player, payoff, [None] * self.n)

    def insert(self, coalition, payoff):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                curr.children[player] = self.get_node(player, 0)
            curr = curr.children[player]
        curr.payoff = payoff

    def search(self, coalition):
        curr = self.root
        for player in coalition:
            if curr.children[player] is None:
                return None
            curr = curr.children[player]
        return curr.payoff

    def visualize(self):
        self._visualize(self.root, 0)

    def _visualize(self, node, level):
        if node:
            print(f"{'-'*level}{node.player}:{node.payoff}")
            for child in node.children:
                self._visualize(child, level + 1)


def superadditive_extension(n, v):
    """
    n (int): number of players
    v (list of floats): dim = 2 ** n - 1, each entry is a payoff
    """
    coalition_sets = get_ordered_coalitions(n)
    coalition_lists = [sorted(list(c)) for c in coalition_sets]
    coalition_trie = CoalitionTrie(coalition_lists, n, v)
    v_ext = v[:]
    for i, coalition in enumerate(coalition_lists):
        partition_payoff = []
        for part in set_partitions(coalition, 2):
            subpart_payoff = []
            for subpart in part:
                subpart_payoff += [coalition_trie.search(subpart)]
            partition_payoff += [sum(subpart_payoff)]
        v_ext[i] = max(partition_payoff + [v[i]])
        coalition_trie.insert(coalition, v_ext[i])
    return v_ext


"""
Evaluation functions
"""


def scores2coalition(scores, sparsity):
    scores_tensor = torch.tensor(scores)
    top_idx = scores_tensor.argsort(descending=True).tolist()
    cutoff = int(len(scores) * (1 - sparsity))
    cutoff = min(cutoff, (scores_tensor > 0).sum().item())
    coalition = top_idx[:cutoff]
    return coalition


def evaluate_coalition(explainer, data, coalition):
    device = explainer.device
    data = data.to(device)
    pred_prob = explainer.model(data).softmax(dim=-1)
    target_class = pred_prob.argmax(-1).item()
    original_prob = pred_prob[:, target_class].item()

    num_nodes = data.num_nodes
    if len(coalition) == num_nodes:
        # Edge case: pick the graph itself as the explanation, for synthetic data
        masked_prob = original_prob
        maskout_prob = 0
    elif len(coalition) == 0:
        # Edge case: pick the empty set as the explanation, for synthetic data
        masked_prob = 0
        maskout_prob = original_prob
    else:
        mask = torch.zeros(num_nodes).type(torch.float32).to(device)
        mask[coalition] = 1.0
        masked_x, masked_edge_index = explainer.subgraph_building_func(
            data.x.float(), data.edge_index, mask
        )
        masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device)
        masked_prob = (
            explainer.model(masked_data).softmax(dim=-1)[:, target_class].item()
        )

        maskout_x, maskout_edge_index = explainer.subgraph_building_func(
            data.x.float(), data.edge_index, 1 - mask
        )
        maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
        maskout_prob = (
            explainer.model(maskout_data).softmax(dim=-1)[:, target_class].item()
        )

    fidelity = original_prob - maskout_prob
    inv_fidelity = original_prob - masked_prob
    sparsity = 1 - len(coalition) / num_nodes
    return fidelity, inv_fidelity, sparsity


def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
    """
    The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

    Step1: normalize both scores with sparsity
        norm_fidelity = fidelity * sparsity
        norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    Step2: rescale both normalized scores from [-1, 1] to [0, 1]
        rescaled_fidelity = (1 + norm_fidelity) / 2
        rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
    Step3: take the harmonic mean of two rescaled scores
        2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

    Simplifying these three steps gives the formula
    """
    norm_fidelity = fidelity * sparsity
    norm_inv_fidelity = inv_fidelity * (1 - sparsity)
    harmonic_fidelity = (
        (1 + norm_fidelity)
        * (1 - norm_inv_fidelity)
        / (2 + norm_fidelity - norm_inv_fidelity)
    )
    return norm_fidelity, norm_inv_fidelity, harmonic_fidelity


def evaluate_scores_list(explainer, data_list, scores_list, sparsity, logger=None):
    """
    Evaluate the node importance scoring methods, where each node has an associated score,
    i.e. GStarX and GraphSVX.

    Args:
    data_list (list of PyG data)
    scores_list (list of lists): each entry is a list with scores of nodes in a graph

    """

    assert len(data_list) == len(scores_list)

    f_list = []
    inv_f_list = []
    n_f_list = []
    n_inv_f_list = []
    sp_list = []
    h_f_list = []
    for i, data in enumerate(data_list):
        node_scores = scores_list[i]
        coalition = scores2coalition(node_scores, sparsity)
        f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
        n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)

        f_list += [f]
        inv_f_list += [inv_f]
        n_f_list += [n_f]
        n_inv_f_list += [n_inv_f]
        sp_list += [sp]
        h_f_list += [h_f]

    f_mean = np.mean(f_list).item()
    inv_f_mean = np.mean(inv_f_list).item()
    n_f_mean = np.mean(n_f_list).item()
    n_inv_f_mean = np.mean(n_inv_f_list).item()
    sp_mean = np.mean(sp_list).item()
    h_f_mean = np.mean(h_f_list).item()

    if logger is not None:
        logger.info(
            f"Fidelity Mean: {f_mean:.4f}\n"
            f"Inv-Fidelity Mean: {inv_f_mean:.4f}\n"
            f"Norm-Fidelity Mean: {n_f_mean:.4f}\n"
            f"Norm-Inv-Fidelity Mean: {n_inv_f_mean:.4f}\n"
            f"Sparsity Mean: {sp_mean:.4f}\n"
            f"Harmonic-Fidelity Mean: {h_f_mean:.4f}\n"
        )

    return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean


"""
Visualization
"""


def coalition2subgraph(coalition, data, relabel_nodes=True):
    sub_data = copy.deepcopy(data)
    node_mask = torch.zeros(data.num_nodes)
    node_mask[coalition] = 1

    sub_data.x = data.x[node_mask == 1]
    sub_data.edge_index, _ = subgraph(
        node_mask.bool(), data.edge_index, relabel_nodes=relabel_nodes
    )
    return sub_data


def to_networkx(
    data,
    node_index=None,
    node_attrs=None,
    edge_attrs=None,
    to_undirected=False,
    remove_self_loops=False,
):
    r"""
    Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids

    Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)


        node_index (iterable): Pass in it when there are some nodes missing.
                 max(node_index) == max(data.edge_index)
                 len(node_index) == data.num_nodes
    """
    import networkx as nx

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    if node_index is not None:
        """
        There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
        """
        G.add_nodes_from(node_index)
    else:
        G.add_nodes_from(range(data.num_nodes))

    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


"""
Adapted from SubgraphX DIG implementation
https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

Slightly modified the molecule drawing args
"""


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """plot function for different dataset"""
        if self.dataset_name.lower() in ["ba_2motifs"]:
            self.plot_ba2motifs(
                graph, nodelist, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["mutag", "bbbp", "bace"]:
            x = kwargs.get("x")
            self.plot_molecule(
                graph, nodelist, x, title_sentence=title_sentence, figname=figname
            )
        elif self.dataset_name.lower() in ["graph_sst2", "twitter"]:
            words = kwargs.get("words")
            self.plot_sentence(
                graph,
                nodelist,
                words=words,
                title_sentence=title_sentence,
                figname=figname,
            )
        else:
            raise NotImplementedError

    def plot_subgraph(
        self,
        graph,
        nodelist,
        colors: Union[None, str, List[str]] = "#FFA500",
        labels=None,
        edge_color="gray",
        edgelist=None,
        subgraph_edge_color="black",
        title_sentence=None,
        figname=None,
    ):

        if edgelist is None:
            edgelist = [
                (n_frm, n_to)
                for (n_frm, n_to) in graph.edges()
                if n_frm in nodelist and n_to in nodelist
            ]

        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(
            graph,
            pos_nodelist,
            nodelist=nodelist,
            node_color="black",
            node_shape="o",
            node_size=400,
        )
        nx.draw_networkx_nodes(
            graph, pos, nodelist=list(graph.nodes()), node_color=colors, node_size=200
        )
        nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, arrows=False)
        nx.draw_networkx_edges(
            graph,
            pos=pos_nodelist,
            edgelist=edgelist,
            width=6,
            edge_color="black",
            arrows=False,
        )

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis("off")
        if title_sentence is not None:
            plt.title(
                "\n".join(wrap(title_sentence, width=60)), fontdict={"fontsize": 15}
            )
        if figname is not None:
            plt.savefig(figname, format=figname[-3:])

        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(
        self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None
    ):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(
                graph,
                pos_coalition,
                nodelist=nodelist,
                node_color="yellow",
                node_shape="o",
                node_size=500,
            )
            if edgelist is None:
                edgelist = [
                    (n_frm, n_to)
                    for (n_frm, n_to) in graph.edges()
                    if n_frm in nodelist and n_to in nodelist
                ]
                nx.draw_networkx_edges(
                    graph,
                    pos=pos_coalition,
                    edgelist=edgelist,
                    width=5,
                    edge_color="yellow",
                )

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color="grey")
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis("off")
        plt.title("\n".join(wrap(" ".join(words), width=50)))
        if title_sentence is not None:
            string = "\n".join(wrap(" ".join(words), width=50)) + "\n"
            string += "\n".join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(
        self, graph, nodelist, edgelist=None, title_sentence=None, figname=None
    ):
        return self.plot_subgraph(
            graph,
            nodelist,
            edgelist=edgelist,
            title_sentence=title_sentence,
            figname=figname,
        )

    def plot_molecule(
        self, graph, nodelist, x, edgelist=None, title_sentence=None, figname=None
    ):
        # collect the text information and node color
        if self.dataset_name == "mutag":
            node_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
            node_idxs = {
                k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])
            }
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = [
                "#E49D1C",
                "#4970C6",
                "#FF5357",
                "#29A329",
                "brown",
                "darkslategray",
                "#F0EA00",
            ]
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in ["bbbp", "bace"]:
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {
                k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                for k, v in element_idxs.items()
            }
            node_color = [
                "#29A329",
                "lime",
                "#F0EA00",
                "maroon",
                "brown",
                "#E49D1C",
                "#4970C6",
                "#FF5357",
            ]
            colors = [
                node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()
            ]
        else:
            raise NotImplementedError

        self.plot_subgraph(
            graph,
            nodelist,
            colors=colors,
            labels=node_labels,
            edgelist=edgelist,
            edge_color="gray",
            subgraph_edge_color="black",
            title_sentence=title_sentence,
            figname=figname,
        )




class GStarX(object):
    def __init__(
        self,
        model,
        device,
        max_sample_size=5,
        tau=0.01,
        payoff_type="norm_prob",
        payoff_avg=None,
        subgraph_building_method="remove",
    ):

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.max_sample_size = max_sample_size
        self.coalitions = get_ordered_coalitions(max_sample_size)
        self.tau = tau
        self.M = get_associated_game_matrix_M(self.coalitions, max_sample_size, tau)
        self.M = self.M.to(device)

        self.payoff_type = payoff_type
        self.payoff_avg = payoff_avg
        self.subgraph_building_func = get_graph_build_func(subgraph_building_method)

    def explain(
        self, data, superadditive_ext=True, sample_method="khop", num_samples=10, k=3
    ):
        """
        Args:
        sample_method (str): `khop` or `random`. see `sample_subgraph` in utils for details
        num_samples (int): set to -1 then data.num_nodes will be used as num_samples
        """
        data = data.to(self.device)
        adj = (
            to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            .detach()
            .cpu()
        )
        target_class = self.model(data.x.float(), data.edge_index).argmax(-1).item()
        char_func = get_char_func(
            self.model, target_class, self.payoff_type, self.payoff_avg
        )
        if data.num_nodes < self.max_sample_size:
            scores = self.compute_scores(data, adj, char_func, superadditive_ext)
        else:
            scores = torch.zeros(data.num_nodes)
            counts = torch.zeros(data.num_nodes)
            if sample_method == "khop" or num_samples == -1:
                num_samples = data.num_nodes

            i = 0
            while not counts.all() or i < num_samples:
                sampled_nodes, sampled_data, sampled_adj = sample_subgraph(
                    data, self.max_sample_size, sample_method, i, k, adj
                )
                sampled_scores = self.compute_scores(
                    sampled_data, sampled_adj, char_func, superadditive_ext
                )
                scores[sampled_nodes] += sampled_scores
                counts[sampled_nodes] += 1
                i += 1

            nonzero_mask = counts != 0
            scores[nonzero_mask] = scores[nonzero_mask] / counts[nonzero_mask]
        return scores.tolist()

    def compute_scores(self, data, adj, char_func, superadditive_ext=True):
        n = data.num_nodes
        if n == self.max_sample_size:  # use pre-computed results
            coalitions = self.coalitions
            M = self.M
        else:
            coalitions = get_ordered_coalitions(n)
            M = get_associated_game_matrix_M(coalitions, n, self.tau)
            M = M.to(self.device)

        v = get_coalition_payoffs(
            data, coalitions, char_func, self.subgraph_building_func
        )
        if superadditive_ext:
            v = v.tolist()
            v_ext = superadditive_extension(n, v)
            v = torch.tensor(v_ext).to(self.device)

        P = get_associated_game_matrix_P(coalitions, n, adj)
        P = P.to(self.device)
        H = torch.sparse.mm(P, torch.sparse.mm(M, P))
        H_tilde = get_limit_game_matrix(H, is_sparse=True)
        v_tilde = torch.sparse.mm(H_tilde, v.view(-1, 1)).view(-1)
    
        scores = v_tilde[:n].cpu()
        return scores


# In[31]:


preds = []
for data in test_dataset:
    try:
        data.to(device)
        data.x = data.x.float()
        pred = model(data.x.float(), data.edge_index).softmax(-1)
        preds += [pred]
    except: pass
preds = torch.concat(preds)
payoff_avg = preds.mean(0).tolist()
gstarx = GStarX(model, device, payoff_avg=payoff_avg)
gstarx_explainer_results = []

for data in tqdm(test_dataset):
    try:
        data = data.to(device)
        data.x = data.x.float()
        r = {
            'data': data,
            'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
            'res':{}
        }
        soft_mask = torch.tensor(gstarx.explain(data, superadditive_ext=False, num_samples=5))
        r['soft_mask'] = soft_mask
        hard_masks = generate_hard_masks(soft_mask)
        for sparsity, hard_mask in hard_masks:
            # print(sparsity)
            r['res'][sparsity.item()] = calculate_fidelity(data, hard_mask, model)
            r['res'][sparsity.item()]['hard_mask'] = hard_mask
        r['res_topk'] = {
            1: calculate_fidelity_topk(data, soft_mask, model,1),
            3: calculate_fidelity_topk(data, soft_mask, model,3),
            5: calculate_fidelity_topk(data, soft_mask, model,5)
        }
        gstarx_explainer_results.append(r)
    except:
        pass


# In[32]:


with open(f'post_hoc/{dataset_name}/{seed}/gstarx.pkl', 'wb') as f:
    pickle.dump(gstarx_explainer_results, f)
gstarx_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/gstarx.pkl', 'rb'))


# # SubGraphX

# In[34]:


from graphxai.explainers import SubgraphX
subgraphx_explainer = SubgraphX(model, sample_num=10)

subgraphx_explainer_results = []
for data in tqdm(test_dataset):
    try:
        data = data.to(device)
        data.x = data.x.float()
        r = {
            'data': data,
            'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
            'res':{}
        }
        exp = subgraphx_explainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index, label=torch.tensor(r['pred'].argmax(-1)))
        soft_mask = exp.node_imp
        r['soft_mask'] = soft_mask
        hard_masks = generate_hard_masks(soft_mask)
        for sparsity, hard_mask in hard_masks:
            # print(sparsity)
            r['res'][sparsity.item()] = calculate_fidelity(data, hard_mask, model)
            r['res'][sparsity.item()]['hard_mask'] = hard_mask
        r['res_topk'] = {
            1: calculate_fidelity_topk(data, soft_mask, model,1),
            3: calculate_fidelity_topk(data, soft_mask, model,3),
            5: calculate_fidelity_topk(data, soft_mask, model,5)
        }
    except: 
        continue
    subgraphx_explainer_results.append(r)


# In[35]:


with open(f'post_hoc/{dataset_name}/{seed}/subgraphx.pkl', 'wb') as f:
    pickle.dump(subgraphx_explainer_results, f)
subgraphx_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/subgraphx.pkl', 'rb'))




# def train_epoch(model_tell, loader, device, optimizer, num_classes, reg=1, sqrt_reg=False):
#     model_tell.train()
    
#     total_loss = 0
#     total_correct = 0
    
#     for data in loader:
#         try:
#             loss = 0
#             if data.x is None:
#                 data.x = torch.ones((data.num_nodes, model_tell.num_features))
#             if data.y.numel() == 0: continue
#             if data.x.isnan().any(): continue
#             if data.y.isnan().any(): continue
#             data.x = data.x.float()
#             y = data.y.reshape(-1).to(device).long()
#             optimizer.zero_grad()

#             out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))       
#             pred = out.argmax(-1)
#             loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
#             # loss += reg*(torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean() + model_tell.fc.phi_in.entropy)
#             loss += reg*model_tell.fc.phi_in.entropy
#             if sqrt_reg:
#                 loss+= reg*torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean()
#             else:
#                 loss+=reg*model_tell.fc.reg_loss
#             loss.backward()
#             zero_nan_gradients(model_tell)#torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean()  + 
#             optimizer.step()
#             total_loss += loss.item() * data.num_graphs / len(loader.dataset)
#             total_correct += pred.eq(y).sum().item() / len(loader.dataset)
#         except Exception as e:
#             print(e)
#             pass

#     return total_loss, total_correct

# model_tell = GIN(num_classes=num_classes, num_features=num_features, num_layers=num_layers, hidden_dim=hidden_dim, nogumbel=True)
# model_tell.load_state_dict(torch.load(os.path.join(results_path, 'best.pt'), map_location=device))
# model_tell = model_tell.to(device)



# model_tell.fc = LogicalLayer(model_tell.fc1.in_features, num_classes).to(device)
# model_tell.fc.phi_in.tau = 10

# def forward_tell(self):
#     def fwd(x, edge_index, batch=None, activations=False, *args, **kwargs):
#         if batch is None:
#             batch = torch.zeros(x.shape[0]).long().to(x.device)
#         xs = []
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             xs.append(x)
#             x = self.dropout(x)
    
#         x_mean = global_mean_pool(torch.hstack(xs), batch)
#         x_max = global_max_pool(torch.hstack(xs), batch)
#         x_sum = global_add_pool(torch.hstack(xs), batch)
#         x = torch.hstack([x_mean, x_max, x_sum])
#         # x = self.dropout(x)
#         acts = self.fc.phi_in(x)
#         x = self.fc(x)
#         if activations:
#             return x, acts, xs
#         return x
#     return fwd

# #TOREMOVE

# model_tell = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/tell_model.pkl', 'rb'))


# #UNTIL HERE


# model_tell.forward = forward_tell(model_tell)
# model_tell.fc.phi_in.w.shape
# optimizer = torch.optim.Adam(model_tell.fc.parameters(), lr=0.001, weight_decay=0)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# for i in range(2000):
#     train_loss, train_acc = train_epoch(model_tell, train_loader, device, optimizer, num_classes, reg=0.1 if i<=800 else 0.01, sqrt_reg=i>800)
#     val_acc = test_epoch(model_tell, val_loader, device)
#     test_acc = test_epoch(model_tell, test_loader, device)
#     if i%10 == 0:
#         print(i, train_loss, train_acc, val_acc, test_acc, (model_tell.fc.weight>1e-4).sum())


# model_tell.forward = forward_tell(model_tell)


# from torch_geometric.utils import k_hop_subgraph
# feat_map = []
# for readout in ['mean', 'max', 'sum']:
#     for l in range(num_layers):
#         for d in range(hidden_dim):
#             feat_map.append((readout, l, d))

# tell_explainer_results = []
# for data in tqdm(test_dataset):
#     try:
#         data = data.to(device)
#         data.x = data.x.float()
#         pred_tell, rule_acts, layers_acts = model_tell(data.x.float(), data.edge_index, activations=True)
        
#         pred = model(data.x.float(), data.edge_index)
#         # rule_acts = rule_acts>0.5
#         r = {
#             'data': data,
#             'pred': pred.softmax(-1).detach().cpu().numpy(),
#             'res':{}
#         }
#         pred_c = r['pred'].argmax(-1).item()
#         rules = extract_rules(model_tell.fc)
#         soft_mask = torch.zeros(data.x.shape[0]).to(device)
#         for c, class_rules in enumerate(rules):
#             for rule in class_rules:
#                 # print(rule)
#                 # print(rule_acts[:,rule])
                
#                 # if not rule_activated: continue
#                 for literal in rule:
#                     # print(literal)
#                     agg, layer, i = feat_map[literal]
#                     acts = layers_acts[layer][:,i]
#                     m = torch.zeros_like(soft_mask)
#                     if agg == 'max':
#                         m[acts>=acts.max()] = (1 if pred_c==c else -1)*acts.max()*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
#                     elif agg == 'sum':
#                         m=(1 if pred_c==c else -1)*acts*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
#                     else:
#                         m=(1 if pred_c==c else -1)*acts*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
#                     m_=torch.zeros_like(m)
#                     for i in range(len(m)):
#                         if m[i] > 0:
#                             try:
#                                 subset, _, _, _ = k_hop_subgraph(i, 1, data.edge_index.cpu())
#                                 m_[subset] += m[i]
#                             except:
#                                 m_[i] = m[i]
#                     soft_mask+=m_    

#         # print(soft_mask)
#         soft_mask = soft_mask.detach().cpu()
#         r['soft_mask'] = soft_mask
#         hard_masks = generate_hard_masks(soft_mask)
#         for sparsity, hard_mask in hard_masks:
#             sparsity = sparsity.item()
#             r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
#             r['res'][sparsity]['hard_mask'] = hard_mask
#         r['res_topk'] = {
#             1: calculate_fidelity_topk(data, soft_mask, model,1),
#             3: calculate_fidelity_topk(data, soft_mask, model,3),
#             5: calculate_fidelity_topk(data, soft_mask, model,5)
#         }
#         tell_explainer_results.append(r)
#     except Exception as e:
#         print(e)


# model_tell.forward = None

# with open(f'post_hoc/{dataset_name}/{seed}/tell_model.pkl', 'wb') as f:
#     pickle.dump(model_tell, f)


# with open(f'post_hoc/{dataset_name}/{seed}/tell.pkl', 'wb') as f:
#     pickle.dump(tell_explainer_results, f)
# torch.save(model_tell, f'post_hoc/{dataset_name}/{seed}/model_tell.pt')
# tell_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/tell.pkl', 'rb'))



######FINE#######




# torch.no_grad()
# def find_logic_rules(w, t_in, t_out, activations=None, max_rule_len=10, max_rules=100, min_support=1):
#     w = w.clone()
#     t_in = t_in.clone()
#     t_out = t_out.clone()
#     t_out = t_out.item()
#     ordering_scores = w
#     sorted_idxs = torch.argsort(ordering_scores, 0, descending=True)
#     mask = w > 1e-5
#     if activations is not None:
#         mask = mask & (activations.sum(0) >= min_support)
#     total_result = set()

#     # Filter and sort indices based on the mask
#     idxs_to_visit = sorted_idxs[mask[sorted_idxs]]
#     if idxs_to_visit.numel() == 0:
#         return total_result

#     # Sort weights based on the filtered indices
#     sorted_weights = w[idxs_to_visit]
#     current_combination = []
#     result = set()

#     def find_logic_rules_recursive(index, current_sum):
#         # Stop if the maximum number of rules has been reached
#         if len(result) >= max_rules:
#             return

#         if len(current_combination) > max_rule_len:
#             return

#         # Check if the current combination satisfies the condition
#         if current_sum >= t_out:
#             c = idxs_to_visit[current_combination].cpu().detach().tolist()
#             c = tuple(sorted(c))
#             result.add(c)
#             return

#         # Prune if remaining weights can't satisfy t_out
#         remaining_max_sum = current_sum + sorted_weights[index:].sum()
#         if remaining_max_sum < t_out:
#             return

#         # Explore further combinations
#         for i in range(index, idxs_to_visit.shape[0]):
#             # Prune based on activations if provided
#             if activations is not None and len(current_combination) > 0 and activations[:, idxs_to_visit[current_combination + [i]]].all(-1).sum().item() < min_support:
#                 continue

#             current_combination.append(i)
#             find_logic_rules_recursive(i + 1, current_sum + sorted_weights[i])
#             current_combination.pop()

#     # Start the recursive process
#     find_logic_rules_recursive(0, 0)
#     return result


# def extract_rules(self, feature=None, activations=None, max_rule_len=float('inf'), max_rules=100, min_support=1, out_threshold=0.5):
#     ws = self.weight
#     t_in = self.phi_in.t
#     t_out = -self.b + inverse_sigmoid(torch.tensor(out_threshold))

#     rules = []
#     if feature is None:
#         features = range(self.out_features)
#     else:
#         features = [feature]
#     for i in features:
#         w = ws[i].to('cpu')
#         ti = t_in.to('cpu')
#         to = t_out[i].to('cpu')
#         rules.append(find_logic_rules(w, ti, to, activations, max_rule_len, max_rules, min_support))

#     return rules



# def get_blues_color(value):
#     cmap = plt.get_cmap("Blues")  # Get the Blues colormap
#     return cmap(value)  # Return the RGBA color

# def plot_activations(batch_ids, batch, attr, soft=False):
#     if type(batch_ids) != list:
#         batch_ids = [batch_ids]
#     if soft: attr = (attr-attr.min() + 1e-6)/(attr.max()-attr.min()+1e-6)
#     fig, axs = plt.subplots(1, len(batch_ids), figsize=(16*len(batch_ids), 8))
#     if type(axs) != np.ndarray: axs = np.array([axs])
#     for i, batch_id in enumerate(batch_ids):
#         node_mask = batch.batch == batch_id  # Get nodes where batch == 0
#         node_indices = torch.nonzero(node_mask, as_tuple=True)[0]
        
#         subgraph_edge_mask = (batch.batch[batch.edge_index[0]] == batch_id) & \
#                              (batch.batch[batch.edge_index[1]] == batch_id)
#         subgraph_edges = batch.edge_index[:, subgraph_edge_mask]
        
#         node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
#         remapped_edges = torch.tensor([[node_mapping[e.item()] for e in edge] for edge in subgraph_edges.T])
        
#         G = nx.Graph()
#         G.add_edges_from(remapped_edges.numpy())
        
#         nx.set_node_attributes(G, {v: k for k, v in node_mapping.items()}, "original_id")
        
#         node_colors = []
#         node_borders = []
        
#         for node in G.nodes:
#             node_colors.append(get_blues_color(attr[batch.batch==batch_id][node]))  # Fill color
            
        
#         pos = nx.kamada_kawai_layout(G) 
        
#         nx.draw(
#             G, pos,
#             node_color=node_colors,
#             edgecolors=node_borders,  # Border colors
#             node_size=700,
#             with_labels=False,
#             ax = axs[i]
#         )
        
#         axs[i].set_title(f"Class = {batch.y[batch_id]}")
#     plt.show()



# import torch
# import torch.nn.functional as F
# from torch_geometric.utils import subgraph

# def calculate_fidelity(data, node_mask, model, remove_nodes=True, top_k=None):

#     device = next(model.parameters()).device
#     data = data.to(device)

#     # Compute sparsity
#     total_nodes = data.x.shape[0]
#     sparsity = 1 - node_mask.sum().item() / total_nodes if total_nodes > 0 else 0

#     # Get original predictions
#     original_pred = model(data.x.float(), data.edge_index)
#     # original_pred = F.softmax(original_pred, dim=1)
#     label = original_pred.argmax(-1).item()
    
#     if remove_nodes:
#         masked_edge_index, _ = subgraph(node_mask == 0, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
#         masked_pred = model(data.x[node_mask == 0], masked_edge_index)
#     else:
#         masked_x = data.x.clone()
#         masked_x[node_mask==1] = 0
#         masked_pred = model(masked_x, data.edge_index)

#     if remove_nodes:
#         masked_edge_index, _ = subgraph(node_mask == 1, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
#         retained_pred = model(data.x[node_mask == 1], masked_edge_index)
#     else:
#         masked_x = data.x.clone()
#         masked_x[node_mask==0] = 0
#         retained_pred = model(masked_x, data.edge_index)
#     # retained_pred = F.softmax(retained_pred, dim=1)


#     # Compute Fidelity+ and Fidelity-
#     # inv_fidelity = (original_pred[:, label] - 
#     #                              retained_pred[:, label]).mean().item()

#     # fidelity = (original_pred[:, label] - 
#     #                              masked_pred[:, label]).mean().item()

#     inv_fidelity = (original_pred.argmax(-1) != 
#                                  retained_pred.argmax(-1)).float().item()

#     fidelity = (original_pred.argmax(-1)  != 
#                                  masked_pred.argmax(-1) ).float().item()

#     n_fidelity = inv_fidelity*sparsity
#     n_inv_fidelity = inv_fidelity*(1-sparsity)
    
#     # Compute HFidelity (harmonic mean of Fidelity+ and Fidelity-)
#     hfidelity = ((1+n_fidelity) * (1-n_inv_fidelity)) / (2 + n_fidelity - n_inv_fidelity) if (1 + n_fidelity - n_inv_fidelity) != 0 else 0

#     return {
#         "Fidelity": fidelity,
#         "InvFidelity": inv_fidelity,
#         "HFidelity": hfidelity
#     }


# def calculate_fidelity_topk(data, node_soft_mask, model, top_k, remove_nodes=True):

#     device = next(model.parameters()).device
#     data = data.to(device)

#     # Compute sparsity
#     total_nodes = data.x.shape[0]

#     # Get original predictions
#     original_pred = model(data.x.float(), data.edge_index)
#     # original_pred = F.softmax(original_pred, dim=1)
#     label = original_pred.argmax(-1).item()

    

#     node_mask = torch.zeros_like(node_soft_mask)
#     node_mask[torch.topk(node_soft_mask, top_k).indices] = 1
    
#     if remove_nodes:
#         masked_edge_index, _ = subgraph(node_mask == 0, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
#         masked_pred = model(data.x[node_mask == 0], masked_edge_index)
#     else:
#         masked_x = data.x.clone()
#         masked_x[node_mask==1] = 0
#         masked_pred = model(masked_x, data.edge_index)

#     node_mask = torch.zeros_like(node_soft_mask)
#     node_mask[torch.topk(node_soft_mask, total_nodes-top_k).indices] = 1
    
#     if remove_nodes:
#         masked_edge_index, _ = subgraph(node_mask == 1, edge_index=data.edge_index, num_nodes=data.x.size(0), relabel_nodes=True)
#         retained_pred = model(data.x[node_mask == 1], masked_edge_index)
#     else:
#         masked_x = data.x.clone()
#         masked_x[node_mask==0] = 0
#         retained_pred = model(masked_x, data.edge_index)


#     # inv_fidelity = (original_pred[:, label] - 
#     #                              retained_pred[:, label]).mean().item()

#     # fidelity = (original_pred[:, label] - 
#     #                              masked_pred[:, label]).mean().item()

#     inv_fidelity = (original_pred.argmax(-1) != 
#                                  retained_pred.argmax(-1)).float().item()

#     fidelity = (original_pred.argmax(-1)  != 
#                                  masked_pred.argmax(-1) ).float().item()
#     return {
#         "Fidelity": fidelity,
#         "InvFidelity": inv_fidelity,
#     }




# # In[105]:


# def node_imp_from_edge_imp(edge_index, n_nodes, edge_imp):
#     node_imp = torch.zeros(n_nodes)
#     for i in range(n_nodes):
#         node_imp[i] = edge_imp[(edge_index[0]==i) | (edge_index[1]==i)].mean()
#     return node_imp
        
# def generate_hard_masks(soft_mask):
#     sparsity_levels = torch.arange(0.5,1, 0.05)
#     hard_masks = []
#     for sparsity in sparsity_levels:
#         threshold = np.percentile(soft_mask, sparsity * 100)
#         hard_mask = (soft_mask > threshold).int()
#         if hard_mask.sum() == 0:
#             hard_mask = (soft_mask > soft_mask.min()).int()
#         hard_masks.append(hard_mask)
#     return list(zip(sparsity_levels, hard_masks))



# os.makedirs(f'post_hoc/{dataset_name}/{seed}', exist_ok=True)


# import numpy as np
# import torch
# import torch.nn as nn
# # import tqdm
# import time

# from typing import Optional
# from torch_geometric.utils import k_hop_subgraph
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, GINConv

# from graphxai.explainers._base import _BaseExplainer
# from graphxai.utils import Explanation, node_mask_from_edge_mask

# device = "cuda" if torch.cuda.is_available() else "cpu"

# class PGExplainer(_BaseExplainer):
#     """
#     PGExplainer

#     Code adapted from DIG
#     """
#     def __init__(self, model: nn.Module, emb_layer_name: str = None,
#                  explain_graph: bool = False,
#                  coeff_size: float = 0.01, coeff_ent: float = 5e-4,
#                  t0: float = 5.0, t1: float = 2.0,
#                  lr: float = 0.003, max_epochs: int = 20, eps: float = 1e-3,
#                  num_hops: int = None, in_channels = None):
#         """
#         Args:
#             model (torch.nn.Module): model on which to make predictions
#                 The output of the model should be unnormalized class score.
#                 For example, last layer = CNConv or Linear.
#             emb_layer_name (str, optional): name of the embedding layer
#                 If not specified, use the last but one layer by default.
#             explain_graph (bool): whether the explanation is graph-level or node-level
#             coeff_size (float): size regularization to constrain the explanation size
#             coeff_ent (float): entropy regularization to constrain the connectivity of explanation
#             t0 (float): the temperature at the first epoch
#             t1 (float): the temperature at the final epoch
#             lr (float): learning rate to train the explanation network
#             max_epochs (int): number of epochs to train the explanation network
#             num_hops (int): number of hops to consider for node-level explanation
#         """
#         super().__init__(model, emb_layer_name)

#         # Parameters for PGExplainer
#         self.explain_graph = explain_graph
#         self.coeff_size = coeff_size
#         self.coeff_ent = coeff_ent
#         self.t0 = t0
#         self.t1 = t1
#         self.lr = lr
#         self.eps = eps
#         self.max_epochs = max_epochs
#         self.num_hops = self.L if num_hops is None else num_hops

#         # Explanation model in PGExplainer

#         mult = 2 # if self.explain_graph else 3

#         # if in_channels is None:
#         #     if isinstance(self.emb_layer, GCNConv):
#         #         in_channels = mult * self.emb_layer.out_channels
#         #     elif isinstance(self.emb_layer, GINConv):
#         #         in_channels = mult * self.emb_layer.nn.out_features
#         #     elif isinstance(self.emb_layer, torch.nn.Linear):
#         #         in_channels = mult * self.emb_layer.out_features
#         #     else:
#         #         fmt_string = 'PGExplainer not implemented for embedding layer of type {}, please provide in_channels directly.'
#         #         raise NotImplementedError(fmt_string.format(type(self.emb_layer)))
#         in_channels = mult*hidden_dim*num_layers
#         self.elayers = nn.ModuleList(
#             [nn.Sequential(
#                 nn.Linear(in_channels, 64),
#                 nn.ReLU()),
#              nn.Linear(64, 1)]).to(device)

#     def __concrete_sample(self, log_alpha: torch.Tensor,
#                           beta: float = 1.0, training: bool = True):
#         """
#         Sample from the instantiation of concrete distribution when training.

#         Returns:
#             training == True: sigmoid((log_alpha + noise) / beta)
#             training == False: sigmoid(log_alpha)
#         """
#         if training:
#             random_noise = torch.rand(log_alpha.shape).to(device)
#             random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
#             gate_inputs = (random_noise + log_alpha) / beta
#             gate_inputs = gate_inputs.sigmoid()
#         else:
#             gate_inputs = log_alpha.sigmoid()

#         return gate_inputs

#     def __emb_to_edge_mask(self, emb: torch.Tensor,
#                            x: torch.Tensor, edge_index: torch.Tensor,
#                            node_idx: int = None,
#                            forward_kwargs: dict = {},
#                            tmp: float = 1.0, training: bool = False):
#         """
#         Compute the edge mask based on embedding.

#         Returns:
#             prob_with_mask (torch.Tensor): the predicted probability with edge_mask
#             edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
#         """
# #        if not self.explain_graph and node_idx is None:
# #            raise ValueError('node_idx should be provided.')

#         with torch.set_grad_enabled(training):
#             # Concat relevant node embeddings
#             # import ipdb; ipdb.set_trace()
#             U, V = edge_index  # edge (u, v), U = (u), V = (v)
#             h1 = emb[U]
#             h2 = emb[V]
#             if self.explain_graph:
#                 h = torch.cat([h1, h2], dim=1)
#             else:
#                 h3 = emb.repeat(h1.shape[0], 1)
#                 h = torch.cat([h1, h2], dim=1)

#             # Calculate the edge weights and set the edge mask
#             for elayer in self.elayers:
#                 h = elayer.to(device)(h)
#             h = h.squeeze()
#             edge_weights = self.__concrete_sample(h, tmp, training)
#             n = emb.shape[0]  # number of nodes
#             mask_sparse = torch.sparse_coo_tensor(
#                 edge_index, edge_weights, (n, n))
#             # Not scalable
#             self.mask_sigmoid = mask_sparse.to_dense()
#             # Undirected graph
#             sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
#             edge_mask = sym_mask[edge_index[0], edge_index[1]]
#             #print('edge_mask', edge_mask)
#             self._set_masks(x, edge_index, edge_mask)

#         # Compute the model prediction with edge mask
#         # with torch.no_grad():
#         #     tester = self.model(x, edge_index)
#         #     print(tester)
#         prob_with_mask = self._predict(x, edge_index,
#                                        forward_kwargs=forward_kwargs,
#                                        return_type='prob')
#         self._clear_masks()

#         return prob_with_mask, edge_mask


#     def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
#         num_nodes, num_edges = x.size(0), edge_index.size(1)
#         graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

#         subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
#             node_idx, self.num_hops, edge_index, relabel_nodes=True,
#             num_nodes=num_nodes, flow=self.__flow__())

#         mapping = {int(v): k for k, v in enumerate(subset)}
#         subgraph = graph.subgraph(subset.tolist())
#         nx.relabel_nodes(subgraph, mapping)

#         x = x[subset]
#         for key, item in kwargs.items():
#             if torch.is_tensor(item) and item.size(0) == num_nodes:
#                 item = item[subset]
#             elif torch.is_tensor(item) and item.size(0) == num_edges:
#                 item = item[edge_mask]
#             kwargs[key] = item
#         y = y[subset]
#         return x, edge_index, y, subset, kwargs


#     def train_explanation_model(self, dataset: Data, forward_kwargs: dict = {}):
#         """
#         Train the explanation model.
#         """
#         optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)

#         def loss_fn(prob: torch.Tensor, ori_pred: int):
#             # Maximize the probability of predicting the label (cross entropy)
#             loss = -torch.log(prob[ori_pred] + 1e-6)
#             # Size regularization
#             edge_mask = self.mask_sigmoid
#             loss += self.coeff_size * torch.sum(edge_mask)
#             # Element-wise entropy regularization
#             # Low entropy implies the mask is close to binary
#             edge_mask = edge_mask * 0.99 + 0.005
#             entropy = - edge_mask * torch.log(edge_mask) \
#                 - (1 - edge_mask) * torch.log(1 - edge_mask)
#             loss += self.coeff_ent * torch.mean(entropy)

#             return loss

#         if self.explain_graph:  # Explain graph-level predictions of multiple graphs
#             # Get the embeddings and predicted labels
#             with torch.no_grad():
#                 dataset_indices = list(range(len(dataset)))
#                 self.model.eval()
#                 emb_dict = {}
#                 ori_pred_dict = {}
#                 for gid in tqdm.tqdm(dataset_indices):
#                     data = dataset[gid].to(device)
#                     pred_label = self._predict(data.x.float(), data.edge_index,
#                                                forward_kwargs=forward_kwargs)
#                     # emb = self._get_embedding(data.x.float(), data.edge_index,
#                     #                           forward_kwargs=forward_kwargs)

#                     xs, ys = self.model.forward_e(data.x.float(), data.edge_index)
#                     emb = torch.hstack(ys[:-1])
#                     # OWEN inserting:
#                     emb_dict[gid] = emb.to(device) # Add embedding to embedding dictionary
#                     ori_pred_dict[gid] = pred_label

#             # Train the mask generator
#             duration = 0.0
#             last_loss = 0.0
#             for epoch in range(self.max_epochs):
#                 loss = 0.0
#                 pred_list = []
#                 tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
#                 self.elayers.train()
#                 optimizer.zero_grad()
#                 tic = time.perf_counter()
#                 for gid in tqdm.tqdm(dataset_indices):
#                     data = dataset[gid].to(device)
#                     prob_with_mask, _ = self.__emb_to_edge_mask(
#                         emb_dict[gid], data.x.float(), data.edge_index,
#                         forward_kwargs=forward_kwargs,
#                         tmp=tmp, training=True)
#                     loss_tmp = loss_fn(prob_with_mask.squeeze(), ori_pred_dict[gid])
#                     loss_tmp.backward()
#                     loss += loss_tmp.item()
#                     pred_label = prob_with_mask.argmax(-1).item()
#                     pred_list.append(pred_label)
#                 optimizer.step()
#                 duration += time.perf_counter() - tic
#                 print(f'Epoch: {epoch} | Loss: {loss}')
#                 if abs(loss - last_loss) < self.eps:
#                     break
#                 last_loss = loss

#         else:  # Explain node-level predictions of a graph
#             data = dataset.to(device)
#             X = data.x
#             EIDX = data.edge_index

#             # Get the predicted labels for training nodes
#             with torch.no_grad():
#                 self.model.eval()
#                 explain_node_index_list = torch.where(data.train_mask)[0].tolist()
#                 # pred_dict = {}
#                 label = self._predict(X, EIDX, forward_kwargs=forward_kwargs)
#                 pred_dict = dict(zip(explain_node_index_list, label[explain_node_index_list]))
#                 # for node_idx in tqdm.tqdm(explain_node_index_list):
#                 #     pred_dict[node_idx] = label[node_idx]

#             # Train the mask generator
#             duration = 0.0
#             last_loss = 0.0
#             x_dict = {}
#             edge_index_dict = {}
#             node_idx_dict = {}
#             emb_dict = {}
#             for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
#                 subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, self.num_hops, EIDX, relabel_nodes=True, num_nodes=data.x.shape[0])
# #                 new_node_index.append(int(torch.where(subset == node_idx)[0]))
#                 x_dict[node_idx] = X[subset].to(device)
#                 edge_index_dict[node_idx] = sub_edge_index.to(device)
#                 emb = self._get_embedding(X[subset], sub_edge_index,forward_kwargs=forward_kwargs)
#                 emb_dict[node_idx] = emb.to(device)
#                 node_idx_dict[node_idx] = int(torch.where(subset==node_idx)[0])

#             for epoch in range(self.max_epochs):
#                 loss = 0.0
#                 optimizer.zero_grad()
#                 tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
#                 self.elayers.train()
#                 tic = time.perf_counter()

#                 for iter_idx, node_idx in tqdm.tqdm(enumerate(x_dict.keys())):
#                     prob_with_mask, _ = self.__emb_to_edge_mask(
#                         emb_dict[node_idx], 
#                         x = x_dict[node_idx], 
#                         edge_index = edge_index_dict[node_idx], 
#                         node_idx = node_idx,
#                         forward_kwargs=forward_kwargs,
#                         tmp=tmp, 
#                         training=True)
#                     loss_tmp = loss_fn(prob_with_mask[node_idx_dict[node_idx]], pred_dict[node_idx])
#                     loss_tmp.backward()
#                     # loss += loss_tmp.item()

#                 optimizer.step()
#                 duration += time.perf_counter() - tic
# #                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
#                 # if abs(loss - last_loss) < self.eps:
#                 #     break
#                 # last_loss = loss

#             print(f"training time is {duration:.5}s")

#     def get_explanation_node(self, node_idx: int, x: torch.Tensor,
#                              edge_index: torch.Tensor, label: torch.Tensor = None,
#                              y = None,
#                              forward_kwargs: dict = {}, **_):
#         """
#         Explain a node prediction.

#         Args:
#             node_idx (int): index of the node to be explained
#             x (torch.Tensor, [n x d]): node features
#             edge_index (torch.Tensor, [2 x m]): edge index of the graph
#             label (torch.Tensor, optional, [n x ...]): labels to explain
#                 If not provided, we use the output of the model.
#             forward_kwargs (dict, optional): additional arguments to model.forward
#                 beyond x and edge_index

#         Returns:
#             exp (dict):
#                 exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
#                 exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
#                 exp['node_imp'] (torch.Tensor, [n]): k-hop node importance
#             khop_info (4-tuple of torch.Tensor):
#                 0. the nodes involved in the subgraph
#                 1. the filtered `edge_index`
#                 2. the mapping from node indices in `node_idx` to their new location
#                 3. the `edge_index` mask indicating which edges were preserved
#         """
#         if self.explain_graph:
#             raise Exception('For graph-level explanations use `get_explanation_graph`.')
#         label = self._predict(x, edge_index) if label is None else label

#         khop_info = _, _, _, sub_edge_mask = \
#             k_hop_subgraph(node_idx, self.num_hops, edge_index,
#                            relabel_nodes=False, num_nodes=x.shape[0])
#         emb = self._get_embedding(x, edge_index, forward_kwargs=forward_kwargs)
#         _, edge_mask = self.__emb_to_edge_mask(
#             emb, x, edge_index, node_idx, forward_kwargs=forward_kwargs,
#             tmp=2, training=False)
#         edge_imp = edge_mask[sub_edge_mask]

#         exp = Explanation(
#             node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1], edge_imp.bool()),
#             edge_imp = edge_imp,
#             node_idx = node_idx
#         )

#         exp.set_enclosing_subgraph(khop_info)

#         return exp

#     def get_explanation_graph(self, x: torch.Tensor, edge_index: torch.Tensor,
#                               label: Optional[torch.Tensor] = None,
#                               y = None,
#                               forward_kwargs: dict = {},
#                               top_k: int = 10):
#         """
#         Explain a whole-graph prediction.

#         Args:
#             x (torch.Tensor, [n x d]): node features
#             edge_index (torch.Tensor, [2 x m]): edge index of the graph
#             label (torch.Tensor, optional, [n x ...]): labels to explain
#                 If not provided, we use the output of the model.
#             forward_kwargs (dict, optional): additional arguments to model.forward
#                 beyond x and edge_index
#             top_k (int): number of edges to include in the edge-importance explanation

#         Returns:
#             exp (dict):
#                 exp['feature_imp'] (torch.Tensor, [d]): feature mask explanation
#                 exp['edge_imp'] (torch.Tensor, [m]): k-hop edge importance
#                 exp['node_imp'] (torch.Tensor, [m]): k-hop node importance

#         """
#         if not self.explain_graph:
#             raise Exception('For node-level explanations use `get_explanation_node`.')

#         label = self._predict(x, edge_index,
#                               forward_kwargs=forward_kwargs) if label is None else label

#         with torch.no_grad():
#             # emb = self._get_embedding(x, edge_index,
#             #                           forward_kwargs=forward_kwargs)
#             xs, ys = self.model.forward_e(data.x.float(), data.edge_index)
#             emb = torch.hstack(ys[:-1])
#             _, edge_mask = self.__emb_to_edge_mask(
#                 emb, x, edge_index, forward_kwargs=forward_kwargs,
#                 tmp=1.0, training=False)

#         #exp['edge_imp'] = edge_mask

#         exp = Explanation(
#             node_imp = node_mask_from_edge_mask(torch.arange(x.shape[0], device=x.device), edge_index),
#             edge_imp = edge_mask
#         )

#         exp.set_whole_graph(Data(x=x, edge_index=edge_index))

#         return exp

#     def get_explanation_link(self):
#         """
#         Explain a link prediction.
#         """
#         raise NotImplementedError()


# from concurrent.futures import ThreadPoolExecutor

# def explain_pg_explainer(model, data):
#     r = {
#         'data': data,
#         'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
#         'res' : {}
#     }
#     data = data.to(device)
#     model=model.to(device)
#     pgexplainer = PGExplainer(model, explain_graph=True)
#     exp = pgexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index)
#     soft_mask = node_imp_from_edge_imp(data.edge_index, data.x.shape[0], exp.edge_imp)
#     soft_mask[soft_mask!=soft_mask]=0
#     r['soft_mask'] = soft_mask
#     hard_masks = generate_hard_masks(soft_mask)
#     for sparsity, hard_mask in hard_masks:
#         sparsity=sparsity.item()
#         r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
#         r['res'][sparsity]['hard_mask'] = hard_mask

#     r['res_topk'] = {
#         1: calculate_fidelity_topk(data, soft_mask, model,1),
#         3: calculate_fidelity_topk(data, soft_mask, model,3),
#         5: calculate_fidelity_topk(data, soft_mask, model,5)
#     }
#     return r
    
# pg_explainer_results = []
# for data in tqdm(test_dataset):
#     try:
#         data = data.to(device)
#         data.x = data.x.float()
#         r = explain_pg_explainer(model, data)
#         pg_explainer_results.append(r)
#     except: pass


# with open(f'post_hoc/{dataset_name}/{seed}/pgexplainer.pkl', 'wb') as f:
#     pickle.dump(pg_explainer_results, f)
# pg_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/pgexplainer.pkl', 'rb'))






# from graphxai.explainers import GNNExplainer

# def explain_gnn_explainer(model, data):
#     r = {
#         'data': data,
#         'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
#         'res' : {}
#     }
#     data = data.to(device)
#     model=model.to(device)
#     gnnexplainer = GNNExplainer(model)
#     exp = gnnexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index)
#     soft_mask = node_imp_from_edge_imp(data.edge_index, data.x.shape[0], exp.edge_imp)
#     soft_mask[soft_mask!=soft_mask]=0
#     r['soft_mask'] = soft_mask
#     hard_masks = generate_hard_masks(soft_mask)
#     for sparsity, hard_mask in hard_masks:
#         sparsity=sparsity.item()
#         r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
#         r['res'][sparsity]['hard_mask'] = hard_mask

#     r['res_topk'] = {
#         1: calculate_fidelity_topk(data, soft_mask, model,1),
#         3: calculate_fidelity_topk(data, soft_mask, model,3),
#         5: calculate_fidelity_topk(data, soft_mask, model,5)
#     }
#     return r
    
# gnn_explainer_results = []
# for data in tqdm(test_dataset):
#     data = data.to(device)
#     data.x = data.x.float()
#     try:
#         r = explain_gnn_explainer(model, data)
#         gnn_explainer_results.append(r)
#     except: pass

# with open(f'post_hoc/{dataset_name}/{seed}/gnnexplainer.pkl', 'wb') as f:
#     pickle.dump(gnn_explainer_results, f)
# gnn_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/gnnexplainer.pkl', 'rb'))



# from graphxai.explainers import IntegratedGradExplainer

# def explain_ig_explainer(model, data):
#     r = {
#         'data': data,
#         'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
#         'res' : {}
#     }
#     data = data.to(device)
#     model=model.to(device)
#     igexplainer = IntegratedGradExplainer(model, torch.nn.CrossEntropyLoss())
#     exp = igexplainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index, label=torch.tensor(r['pred'].argmax(-1)).to(device))
#     soft_mask = exp.node_imp.detach().cpu()
#     r['soft_mask'] = soft_mask
#     hard_masks = generate_hard_masks(soft_mask)
#     for sparsity, hard_mask in hard_masks:
#         sparsity=sparsity.item()
#         r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
#         r['res'][sparsity]['hard_mask'] = hard_mask
#     r['res_topk'] = {
#         1: calculate_fidelity_topk(data, soft_mask, model,1),
#         3: calculate_fidelity_topk(data, soft_mask, model,3),
#         5: calculate_fidelity_topk(data, soft_mask, model,5)
#     }
#     return r
    
# ig_explainer_results = []
# for data in tqdm(test_dataset):
#     data = data.to(device)
#     data.x = data.x.float()
#     try:
#         r = explain_ig_explainer(model, data)
#         ig_explainer_results.append(r)
#     except: pass


# # In[27]:


# with open(f'post_hoc/{dataset_name}/{seed}/ig.pkl', 'wb') as f:
#     pickle.dump(ig_explainer_results, f)
# ig_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/ig.pkl', 'rb'))


# # In[28]:


# from graphxai.explainers import PGExplainer


# # # GStarX

# # In[30]:


# import torch
# from torch_geometric.utils import subgraph, to_dense_adj
# import os
# import pytz
# import logging
# import numpy as np
# import random
# import torch
# import networkx as nx
# import copy
# from rdkit import Chem
# from datetime import datetime
# from torch_geometric.utils import subgraph, to_dense_adj
# from torch_geometric.data import Data, Batch, Dataset, DataLoader

# # For associated game
# from itertools import combinations
# from scipy.sparse.csgraph import connected_components as cc

# def set_partitions(iterable, k=None, min_size=None, max_size=None):
#     """
#     Yield the set partitions of *iterable* into *k* parts. Set partitions are
#     not order-preserving.

#     >>> iterable = 'abc'
#     >>> for part in set_partitions(iterable, 2):
#     ...     print([''.join(p) for p in part])
#     ['a', 'bc']
#     ['ab', 'c']
#     ['b', 'ac']


#     If *k* is not given, every set partition is generated.

#     >>> iterable = 'abc'
#     >>> for part in set_partitions(iterable):
#     ...     print([''.join(p) for p in part])
#     ['abc']
#     ['a', 'bc']
#     ['ab', 'c']
#     ['b', 'ac']
#     ['a', 'b', 'c']

#     if *min_size* and/or *max_size* are given, the minimum and/or maximum size
#     per block in partition is set.

#     >>> iterable = 'abc'
#     >>> for part in set_partitions(iterable, min_size=2):
#     ...     print([''.join(p) for p in part])
#     ['abc']
#     >>> for part in set_partitions(iterable, max_size=2):
#     ...     print([''.join(p) for p in part])
#     ['a', 'bc']
#     ['ab', 'c']
#     ['b', 'ac']
#     ['a', 'b', 'c']

#     """
#     L = list(iterable)
#     n = len(L)
#     if k is not None:
#         if k < 1:
#             raise ValueError(
#                 "Can't partition in a negative or zero number of groups"
#             )
#         elif k > n:
#             return

#     min_size = min_size if min_size is not None else 0
#     max_size = max_size if max_size is not None else n
#     if min_size > max_size:
#         return

#     def set_partitions_helper(L, k):
#         n = len(L)
#         if k == 1:
#             yield [L]
#         elif n == k:
#             yield [[s] for s in L]
#         else:
#             e, *M = L
#             for p in set_partitions_helper(M, k - 1):
#                 yield [[e], *p]
#             for p in set_partitions_helper(M, k):
#                 for i in range(len(p)):
#                     yield p[:i] + [[e] + p[i]] + p[i + 1 :]

#     if k is None:
#         for k in range(1, n + 1):
#             yield from filter(
#                 lambda z: all(min_size <= len(bk) <= max_size for bk in z),
#                 set_partitions_helper(L, k),
#             )
#     else:
#         yield from filter(
#             lambda z: all(min_size <= len(bk) <= max_size for bk in z),
#             set_partitions_helper(L, k),
#         )


# # For visualization
# from typing import Union, List
# from textwrap import wrap
# import matplotlib.pyplot as plt


# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def check_dir(save_dirs):
#     if save_dirs:
#         if os.path.isdir(save_dirs):
#             pass
#         else:
#             os.makedirs(save_dirs)


# def timetz(*args):
#     tz = pytz.timezone("US/Pacific")
#     return datetime.now(tz).timetuple()


# def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
#     check_dir(log_path)

#     tz = pytz.timezone("US/Pacific")
#     logger = logging.getLogger(__name__)
#     logger.propagate = False  # avoid duplicate logging
#     logger.setLevel(log_level)

#     # Clean logger first to avoid duplicated handlers
#     for hdlr in logger.handlers[:]:
#         logger.removeHandler(hdlr)

#     file_handler = logging.FileHandler(os.path.join(log_path, log_file))
#     formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
#     formatter.converter = timetz
#     file_handler.setFormatter(formatter)
#     logger.addHandler(file_handler)

#     if console_log:
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)
#     return logger


# def get_graph_build_func(build_method):
#     if build_method.lower() == "zero_filling":
#         return graph_build_zero_filling
#     elif build_method.lower() == "split":
#         return graph_build_split
#     elif build_method.lower() == "remove":
#         return graph_build_remove
#     else:
#         raise NotImplementedError


# """
# Graph building/Perturbation
# `graph_build_zero_filling` and `graph_build_split` are adapted from the DIG library
# """


# def graph_build_zero_filling(X, edge_index, node_mask: torch.Tensor):
#     """subgraph building through masking the unselected nodes with zero features"""
#     ret_X = X * node_mask.unsqueeze(1)
#     return ret_X, edge_index


# def graph_build_split(X, edge_index, node_mask: torch.Tensor):
#     """subgraph building through spliting the selected nodes from the original graph"""
#     ret_X = X
#     row, col = edge_index
#     edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
#     ret_edge_index = edge_index[:, edge_mask]
#     return ret_X, ret_edge_index


# def graph_build_remove(X, edge_index, node_mask: torch.Tensor):
#     """subgraph building through removing the unselected nodes from the original graph"""
#     ret_X = X[node_mask == 1]
#     ret_edge_index, _ = subgraph(node_mask.bool(), edge_index, relabel_nodes=True)
#     return ret_X, ret_edge_index


# """
# Associated game of the HN value
# Implementated using sparse tensor
# """


# def get_ordered_coalitions(n):
#     coalitions = sum(
#         [[set(c) for c in combinations(range(n), k)] for k in range(1, n + 1)], []
#     )
#     return coalitions


# def get_associated_game_matrix_M(coalitions, n, tau):
#     indices = []
#     values = []
#     for i, s in enumerate(coalitions):
#         for j, t in enumerate(coalitions):
#             if i == j:
#                 indices += [[i, j]]
#                 values += [1 - (n - len(s)) * tau]
#             elif len(s) + 1 == len(t) and s.issubset(t):
#                 indices += [[i, j]]
#                 values += [tau]
#             elif len(t) == 1 and not t.issubset(s):
#                 indices += [[i, j]]
#                 values += [-tau]

#     indices = torch.Tensor(indices).t()
#     size = (2**n - 1, 2**n - 1)
#     M = torch.sparse_coo_tensor(indices, values, size)
#     return M


# def get_associated_game_matrix_P(coalitions, n, adj):
#     indices = []
#     for i, s in enumerate(coalitions):
#         idx_s = torch.LongTensor(list(s))
#         num_cc, labels = cc(adj[idx_s, :][:, idx_s])
#         cc_s = []
#         for k in range(num_cc):
#             cc_idx_s = (labels == k).nonzero()[0]
#             cc_s += [set((idx_s[cc_idx_s]).tolist())]
#         for j, t in enumerate(coalitions):
#             if t in cc_s:
#                 indices += [[i, j]]

#     indices = torch.Tensor(indices).t()
#     values = [1.0] * indices.shape[-1]
#     size = (2**n - 1, 2**n - 1)

#     P = torch.sparse_coo_tensor(indices, values, size)
#     return P


# def get_limit_game_matrix(H, exp_power=7, tol=1e-3, is_sparse=True):
#     """
#     Speed up the power computation by
#     1. Use sparse matrices
#     2. Put all tensors on cuda
#     3. Compute powers exponentially rather than linearly
#         i.e. H -> H^2 -> H^4 -> H^8 -> H^16 -> ...
#     """
#     i = 0
#     diff_norm = tol + 1
#     while i < exp_power and diff_norm > tol:
#         if is_sparse:
#             H_tilde = torch.sparse.mm(H, H)
#         else:
#             H_tilde = torch.mm(H, H)
#         diff_norm = (H_tilde - H).norm()
#         H = H_tilde
#         i += 1
#     return H_tilde


# """
# khop or random sampling to generate subgraphs
# """


# def sample_subgraph(
#     data, max_sample_size, sample_method, target_node=None, k=0, adj=None
# ):
#     if sample_method == "khop":
#         # pick nodes within k-hops of target node. Hop by hop until reach max_sample_size
#         if adj is None:
#             adj = (
#                 to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
#                 .detach()
#                 .cpu()
#             )

#         adj_self_loop = adj + torch.eye(data.num_nodes)
#         k_hop_adj = adj_self_loop
#         sampled_nodes = set()
#         m = max_sample_size
#         l = 0
#         while k > 0 and l < m:
#             k_hop_nodes = k_hop_adj[target_node].nonzero().view(-1).tolist()
#             next_hop_nodes = list(set(k_hop_nodes) - sampled_nodes)
#             sampled_nodes.update(next_hop_nodes[: m - l])
#             l = len(sampled_nodes)
#             k -= 1
#             k_hop_adj = torch.mm(k_hop_adj, adj_self_loop)
#         sampled_nodes = torch.tensor(list(sampled_nodes))

#     elif sample_method == "random":  # randomly pick #max_sample_size nodes
#         sampled_nodes = torch.randperm(data.num_nodes)[:max_sample_size]
#     else:
#         ValueError("Unknown sample method")

#     sampled_x = data.x[sampled_nodes]
#     sampled_edge_index, _ = subgraph(
#         sampled_nodes, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
#     )
#     sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index)
#     sampled_adj = adj[sampled_nodes, :][:, sampled_nodes]

#     return sampled_nodes, sampled_data, sampled_adj


# """
# Payoff computation
# """


# def get_char_func(model, target_class, payoff_type="norm_prob", payoff_avg=None):
#     def char_func(data):
#         with torch.no_grad():
#             logits = model(data.x.float(), data.edge_index)
#             if payoff_type == "raw":
#                 payoff = logits[:, target_class]
#             elif payoff_type == "prob":
#                 payoff = logits.softmax(dim=-1)[:, target_class]
#             elif payoff_type == "norm_prob":
#                 prob = logits.softmax(dim=-1)[:, target_class]
#                 payoff = prob - payoff_avg[target_class]
#             elif payoff_type == "log_prob":
#                 payoff = logits.log_softmax(dim=-1)[:, target_class]
#             else:
#                 raise ValueError("unknown payoff type")
#         return payoff

#     return char_func


# class MaskedDataset(Dataset):
#     def __init__(self, data, mask, subgraph_building_func):
#         super().__init__()

#         self.num_nodes = data.num_nodes
#         self.x = data.x
#         self.edge_index = data.edge_index
#         self.device = data.x.device
#         self.y = data.y

#         if not torch.is_tensor(mask):
#             mask = torch.tensor(mask)

#         self.mask = mask.type(torch.float32).to(self.device)
#         self.subgraph_building_func = subgraph_building_func

#     def __len__(self):
#         return self.mask.shape[0]

#     def __getitem__(self, idx):
#         masked_x, masked_edge_index = self.subgraph_building_func(
#             self.x, self.edge_index, self.mask[idx]
#         )
#         masked_data = Data(x=masked_x, edge_index=masked_edge_index)
#         return masked_data


# def get_coalition_payoffs(data, coalitions, char_func, subgraph_building_func):
#     n = data.num_nodes
#     masks = []
#     for coalition in coalitions:
#         mask = torch.zeros(n)
#         mask[list(coalition)] = 1.0
#         masks += [mask]

#     coalition_mask = torch.stack(masks, axis=0)
#     masked_dataset = MaskedDataset(data, coalition_mask, subgraph_building_func)
#     masked_dataloader = DataLoader(
#         masked_dataset, batch_size=1, shuffle=False, num_workers=0
#     )

#     masked_payoff_list = []
#     for masked_data in masked_dataloader:
#         masked_payoff_list.append(char_func(masked_data))

#     masked_payoffs = torch.cat(masked_payoff_list, dim=0)
#     return masked_payoffs


# """
# Superadditive extension
# """


# class TrieNode:
#     def __init__(self, player, payoff=0, children=[]):
#         self.player = player
#         self.payoff = payoff
#         self.children = children


# class CoalitionTrie:
#     def __init__(self, coalitions, n, v):
#         self.n = n
#         self.root = self.get_node(None, 0)
#         for i, c in enumerate(coalitions):
#             self.insert(c, v[i])

#     def get_node(self, player, payoff):
#         return TrieNode(player, payoff, [None] * self.n)

#     def insert(self, coalition, payoff):
#         curr = self.root
#         for player in coalition:
#             if curr.children[player] is None:
#                 curr.children[player] = self.get_node(player, 0)
#             curr = curr.children[player]
#         curr.payoff = payoff

#     def search(self, coalition):
#         curr = self.root
#         for player in coalition:
#             if curr.children[player] is None:
#                 return None
#             curr = curr.children[player]
#         return curr.payoff

#     def visualize(self):
#         self._visualize(self.root, 0)

#     def _visualize(self, node, level):
#         if node:
#             print(f"{'-'*level}{node.player}:{node.payoff}")
#             for child in node.children:
#                 self._visualize(child, level + 1)


# def superadditive_extension(n, v):
#     """
#     n (int): number of players
#     v (list of floats): dim = 2 ** n - 1, each entry is a payoff
#     """
#     coalition_sets = get_ordered_coalitions(n)
#     coalition_lists = [sorted(list(c)) for c in coalition_sets]
#     coalition_trie = CoalitionTrie(coalition_lists, n, v)
#     v_ext = v[:]
#     for i, coalition in enumerate(coalition_lists):
#         partition_payoff = []
#         for part in set_partitions(coalition, 2):
#             subpart_payoff = []
#             for subpart in part:
#                 subpart_payoff += [coalition_trie.search(subpart)]
#             partition_payoff += [sum(subpart_payoff)]
#         v_ext[i] = max(partition_payoff + [v[i]])
#         coalition_trie.insert(coalition, v_ext[i])
#     return v_ext


# """
# Evaluation functions
# """


# def scores2coalition(scores, sparsity):
#     scores_tensor = torch.tensor(scores)
#     top_idx = scores_tensor.argsort(descending=True).tolist()
#     cutoff = int(len(scores) * (1 - sparsity))
#     cutoff = min(cutoff, (scores_tensor > 0).sum().item())
#     coalition = top_idx[:cutoff]
#     return coalition


# def evaluate_coalition(explainer, data, coalition):
#     device = explainer.device
#     data = data.to(device)
#     pred_prob = explainer.model(data).softmax(dim=-1)
#     target_class = pred_prob.argmax(-1).item()
#     original_prob = pred_prob[:, target_class].item()

#     num_nodes = data.num_nodes
#     if len(coalition) == num_nodes:
#         # Edge case: pick the graph itself as the explanation, for synthetic data
#         masked_prob = original_prob
#         maskout_prob = 0
#     elif len(coalition) == 0:
#         # Edge case: pick the empty set as the explanation, for synthetic data
#         masked_prob = 0
#         maskout_prob = original_prob
#     else:
#         mask = torch.zeros(num_nodes).type(torch.float32).to(device)
#         mask[coalition] = 1.0
#         masked_x, masked_edge_index = explainer.subgraph_building_func(
#             data.x.float(), data.edge_index, mask
#         )
#         masked_data = Data(x=masked_x, edge_index=masked_edge_index).to(device)
#         masked_prob = (
#             explainer.model(masked_data).softmax(dim=-1)[:, target_class].item()
#         )

#         maskout_x, maskout_edge_index = explainer.subgraph_building_func(
#             data.x.float(), data.edge_index, 1 - mask
#         )
#         maskout_data = Data(x=maskout_x, edge_index=maskout_edge_index).to(device)
#         maskout_prob = (
#             explainer.model(maskout_data).softmax(dim=-1)[:, target_class].item()
#         )

#     fidelity = original_prob - maskout_prob
#     inv_fidelity = original_prob - masked_prob
#     sparsity = 1 - len(coalition) / num_nodes
#     return fidelity, inv_fidelity, sparsity


# def fidelity_normalize_and_harmonic_mean(fidelity, inv_fidelity, sparsity):
#     """
#     The idea is similar to the F1 score, two measures are summarized to one through harmonic mean.

#     Step1: normalize both scores with sparsity
#         norm_fidelity = fidelity * sparsity
#         norm_inv_fidelity = inv_fidelity * (1 - sparsity)
#     Step2: rescale both normalized scores from [-1, 1] to [0, 1]
#         rescaled_fidelity = (1 + norm_fidelity) / 2
#         rescaled_inv_fidelity = (1 - norm_inv_fidelity) / 2
#     Step3: take the harmonic mean of two rescaled scores
#         2 / (1/rescaled_fidelity + 1/rescaled_inv_fidelity)

#     Simplifying these three steps gives the formula
#     """
#     norm_fidelity = fidelity * sparsity
#     norm_inv_fidelity = inv_fidelity * (1 - sparsity)
#     harmonic_fidelity = (
#         (1 + norm_fidelity)
#         * (1 - norm_inv_fidelity)
#         / (2 + norm_fidelity - norm_inv_fidelity)
#     )
#     return norm_fidelity, norm_inv_fidelity, harmonic_fidelity


# def evaluate_scores_list(explainer, data_list, scores_list, sparsity, logger=None):
#     """
#     Evaluate the node importance scoring methods, where each node has an associated score,
#     i.e. GStarX and GraphSVX.

#     Args:
#     data_list (list of PyG data)
#     scores_list (list of lists): each entry is a list with scores of nodes in a graph

#     """

#     assert len(data_list) == len(scores_list)

#     f_list = []
#     inv_f_list = []
#     n_f_list = []
#     n_inv_f_list = []
#     sp_list = []
#     h_f_list = []
#     for i, data in enumerate(data_list):
#         node_scores = scores_list[i]
#         coalition = scores2coalition(node_scores, sparsity)
#         f, inv_f, sp = evaluate_coalition(explainer, data, coalition)
#         n_f, n_inv_f, h_f = fidelity_normalize_and_harmonic_mean(f, inv_f, sp)

#         f_list += [f]
#         inv_f_list += [inv_f]
#         n_f_list += [n_f]
#         n_inv_f_list += [n_inv_f]
#         sp_list += [sp]
#         h_f_list += [h_f]

#     f_mean = np.mean(f_list).item()
#     inv_f_mean = np.mean(inv_f_list).item()
#     n_f_mean = np.mean(n_f_list).item()
#     n_inv_f_mean = np.mean(n_inv_f_list).item()
#     sp_mean = np.mean(sp_list).item()
#     h_f_mean = np.mean(h_f_list).item()

#     if logger is not None:
#         logger.info(
#             f"Fidelity Mean: {f_mean:.4f}\n"
#             f"Inv-Fidelity Mean: {inv_f_mean:.4f}\n"
#             f"Norm-Fidelity Mean: {n_f_mean:.4f}\n"
#             f"Norm-Inv-Fidelity Mean: {n_inv_f_mean:.4f}\n"
#             f"Sparsity Mean: {sp_mean:.4f}\n"
#             f"Harmonic-Fidelity Mean: {h_f_mean:.4f}\n"
#         )

#     return sp_mean, f_mean, inv_f_mean, n_f_mean, n_inv_f_mean, h_f_mean


# """
# Visualization
# """


# def coalition2subgraph(coalition, data, relabel_nodes=True):
#     sub_data = copy.deepcopy(data)
#     node_mask = torch.zeros(data.num_nodes)
#     node_mask[coalition] = 1

#     sub_data.x = data.x[node_mask == 1]
#     sub_data.edge_index, _ = subgraph(
#         node_mask.bool(), data.edge_index, relabel_nodes=relabel_nodes
#     )
#     return sub_data


# def to_networkx(
#     data,
#     node_index=None,
#     node_attrs=None,
#     edge_attrs=None,
#     to_undirected=False,
#     remove_self_loops=False,
# ):
#     r"""
#     Extend the PyG to_networkx with extra node_index argument, so subgraphs can be plotted with correct ids

#     Converts a :class:`torch_geometric.data.Data` instance to a
#     :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
#     a directed :obj:`networkx.DiGraph` otherwise.

#     Args:
#         data (torch_geometric.data.Data): The data object.
#         node_attrs (iterable of str, optional): The node attributes to be
#             copied. (default: :obj:`None`)
#         edge_attrs (iterable of str, optional): The edge attributes to be
#             copied. (default: :obj:`None`)
#         to_undirected (bool, optional): If set to :obj:`True`, will return a
#             a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
#             undirected graph will correspond to the upper triangle of the
#             corresponding adjacency matrix. (default: :obj:`False`)
#         remove_self_loops (bool, optional): If set to :obj:`True`, will not
#             include self loops in the resulting graph. (default: :obj:`False`)


#         node_index (iterable): Pass in it when there are some nodes missing.
#                  max(node_index) == max(data.edge_index)
#                  len(node_index) == data.num_nodes
#     """
#     import networkx as nx

#     if to_undirected:
#         G = nx.Graph()
#     else:
#         G = nx.DiGraph()

#     if node_index is not None:
#         """
#         There are some nodes missing. The max(data.edge_index) > data.x.shape[0]
#         """
#         G.add_nodes_from(node_index)
#     else:
#         G.add_nodes_from(range(data.num_nodes))

#     node_attrs, edge_attrs = node_attrs or [], edge_attrs or []

#     values = {}
#     for key, item in data(*(node_attrs + edge_attrs)):
#         if torch.is_tensor(item):
#             values[key] = item.squeeze().tolist()
#         else:
#             values[key] = item
#         if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
#             values[key] = item[0]

#     for i, (u, v) in enumerate(data.edge_index.t().tolist()):

#         if to_undirected and v > u:
#             continue

#         if remove_self_loops and u == v:
#             continue

#         G.add_edge(u, v)

#         for key in edge_attrs:
#             G[u][v][key] = values[key][i]

#     for key in node_attrs:
#         for i, feat_dict in G.nodes(data=True):
#             feat_dict.update({key: values[key][i]})

#     return G


# """
# Adapted from SubgraphX DIG implementation
# https://github.com/divelab/DIG/blob/dig/dig/xgraph/method/subgraphx.py

# Slightly modified the molecule drawing args
# """


# class PlotUtils(object):
#     def __init__(self, dataset_name, is_show=True):
#         self.dataset_name = dataset_name
#         self.is_show = is_show

#     def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
#         """plot function for different dataset"""
#         if self.dataset_name.lower() in ["ba_2motifs"]:
#             self.plot_ba2motifs(
#                 graph, nodelist, title_sentence=title_sentence, figname=figname
#             )
#         elif self.dataset_name.lower() in ["mutag", "bbbp", "bace"]:
#             x = kwargs.get("x")
#             self.plot_molecule(
#                 graph, nodelist, x, title_sentence=title_sentence, figname=figname
#             )
#         elif self.dataset_name.lower() in ["graph_sst2", "twitter"]:
#             words = kwargs.get("words")
#             self.plot_sentence(
#                 graph,
#                 nodelist,
#                 words=words,
#                 title_sentence=title_sentence,
#                 figname=figname,
#             )
#         else:
#             raise NotImplementedError

#     def plot_subgraph(
#         self,
#         graph,
#         nodelist,
#         colors: Union[None, str, List[str]] = "#FFA500",
#         labels=None,
#         edge_color="gray",
#         edgelist=None,
#         subgraph_edge_color="black",
#         title_sentence=None,
#         figname=None,
#     ):

#         if edgelist is None:
#             edgelist = [
#                 (n_frm, n_to)
#                 for (n_frm, n_to) in graph.edges()
#                 if n_frm in nodelist and n_to in nodelist
#             ]

#         pos = nx.kamada_kawai_layout(graph)
#         pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

#         nx.draw_networkx_nodes(
#             graph,
#             pos_nodelist,
#             nodelist=nodelist,
#             node_color="black",
#             node_shape="o",
#             node_size=400,
#         )
#         nx.draw_networkx_nodes(
#             graph, pos, nodelist=list(graph.nodes()), node_color=colors, node_size=200
#         )
#         nx.draw_networkx_edges(graph, pos, width=2, edge_color=edge_color, arrows=False)
#         nx.draw_networkx_edges(
#             graph,
#             pos=pos_nodelist,
#             edgelist=edgelist,
#             width=6,
#             edge_color="black",
#             arrows=False,
#         )

#         if labels is not None:
#             nx.draw_networkx_labels(graph, pos, labels)

#         plt.axis("off")
#         if title_sentence is not None:
#             plt.title(
#                 "\n".join(wrap(title_sentence, width=60)), fontdict={"fontsize": 15}
#             )
#         if figname is not None:
#             plt.savefig(figname, format=figname[-3:])

#         if self.is_show:
#             plt.show()
#         if figname is not None:
#             plt.close()

#     def plot_sentence(
#         self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None
#     ):
#         pos = nx.kamada_kawai_layout(graph)
#         words_dict = {i: words[i] for i in graph.nodes}
#         if nodelist is not None:
#             pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
#             nx.draw_networkx_nodes(
#                 graph,
#                 pos_coalition,
#                 nodelist=nodelist,
#                 node_color="yellow",
#                 node_shape="o",
#                 node_size=500,
#             )
#             if edgelist is None:
#                 edgelist = [
#                     (n_frm, n_to)
#                     for (n_frm, n_to) in graph.edges()
#                     if n_frm in nodelist and n_to in nodelist
#                 ]
#                 nx.draw_networkx_edges(
#                     graph,
#                     pos=pos_coalition,
#                     edgelist=edgelist,
#                     width=5,
#                     edge_color="yellow",
#                 )

#         nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

#         nx.draw_networkx_edges(graph, pos, width=2, edge_color="grey")
#         nx.draw_networkx_labels(graph, pos, words_dict)

#         plt.axis("off")
#         plt.title("\n".join(wrap(" ".join(words), width=50)))
#         if title_sentence is not None:
#             string = "\n".join(wrap(" ".join(words), width=50)) + "\n"
#             string += "\n".join(wrap(title_sentence, width=60))
#             plt.title(string)
#         if figname is not None:
#             plt.savefig(figname)
#         if self.is_show:
#             plt.show()
#         if figname is not None:
#             plt.close()

#     def plot_ba2motifs(
#         self, graph, nodelist, edgelist=None, title_sentence=None, figname=None
#     ):
#         return self.plot_subgraph(
#             graph,
#             nodelist,
#             edgelist=edgelist,
#             title_sentence=title_sentence,
#             figname=figname,
#         )

#     def plot_molecule(
#         self, graph, nodelist, x, edgelist=None, title_sentence=None, figname=None
#     ):
#         # collect the text information and node color
#         if self.dataset_name == "mutag":
#             node_dict = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
#             node_idxs = {
#                 k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])
#             }
#             node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
#             node_color = [
#                 "#E49D1C",
#                 "#4970C6",
#                 "#FF5357",
#                 "#29A329",
#                 "brown",
#                 "darkslategray",
#                 "#F0EA00",
#             ]
#             colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

#         elif self.dataset_name in ["bbbp", "bace"]:
#             element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
#             node_idxs = element_idxs
#             node_labels = {
#                 k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
#                 for k, v in element_idxs.items()
#             }
#             node_color = [
#                 "#29A329",
#                 "lime",
#                 "#F0EA00",
#                 "maroon",
#                 "brown",
#                 "#E49D1C",
#                 "#4970C6",
#                 "#FF5357",
#             ]
#             colors = [
#                 node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()
#             ]
#         else:
#             raise NotImplementedError

#         self.plot_subgraph(
#             graph,
#             nodelist,
#             colors=colors,
#             labels=node_labels,
#             edgelist=edgelist,
#             edge_color="gray",
#             subgraph_edge_color="black",
#             title_sentence=title_sentence,
#             figname=figname,
#         )




# class GStarX(object):
#     def __init__(
#         self,
#         model,
#         device,
#         max_sample_size=5,
#         tau=0.01,
#         payoff_type="norm_prob",
#         payoff_avg=None,
#         subgraph_building_method="remove",
#     ):

#         self.model = model
#         self.device = device
#         self.model.to(device)
#         self.model.eval()

#         self.max_sample_size = max_sample_size
#         self.coalitions = get_ordered_coalitions(max_sample_size)
#         self.tau = tau
#         self.M = get_associated_game_matrix_M(self.coalitions, max_sample_size, tau)
#         self.M = self.M.to(device)

#         self.payoff_type = payoff_type
#         self.payoff_avg = payoff_avg
#         self.subgraph_building_func = get_graph_build_func(subgraph_building_method)

#     def explain(
#         self, data, superadditive_ext=True, sample_method="khop", num_samples=10, k=3
#     ):
#         """
#         Args:
#         sample_method (str): `khop` or `random`. see `sample_subgraph` in utils for details
#         num_samples (int): set to -1 then data.num_nodes will be used as num_samples
#         """
#         data = data.to(self.device)
#         adj = (
#             to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
#             .detach()
#             .cpu()
#         )
#         target_class = self.model(data.x.float(), data.edge_index).argmax(-1).item()
#         char_func = get_char_func(
#             self.model, target_class, self.payoff_type, self.payoff_avg
#         )
#         if data.num_nodes < self.max_sample_size:
#             scores = self.compute_scores(data, adj, char_func, superadditive_ext)
#         else:
#             scores = torch.zeros(data.num_nodes)
#             counts = torch.zeros(data.num_nodes)
#             if sample_method == "khop" or num_samples == -1:
#                 num_samples = data.num_nodes

#             i = 0
#             while not counts.all() or i < num_samples:
#                 sampled_nodes, sampled_data, sampled_adj = sample_subgraph(
#                     data, self.max_sample_size, sample_method, i, k, adj
#                 )
#                 sampled_scores = self.compute_scores(
#                     sampled_data, sampled_adj, char_func, superadditive_ext
#                 )
#                 scores[sampled_nodes] += sampled_scores
#                 counts[sampled_nodes] += 1
#                 i += 1

#             nonzero_mask = counts != 0
#             scores[nonzero_mask] = scores[nonzero_mask] / counts[nonzero_mask]
#         return scores.tolist()

#     def compute_scores(self, data, adj, char_func, superadditive_ext=True):
#         n = data.num_nodes
#         if n == self.max_sample_size:  # use pre-computed results
#             coalitions = self.coalitions
#             M = self.M
#         else:
#             coalitions = get_ordered_coalitions(n)
#             M = get_associated_game_matrix_M(coalitions, n, self.tau)
#             M = M.to(self.device)

#         v = get_coalition_payoffs(
#             data, coalitions, char_func, self.subgraph_building_func
#         )
#         if superadditive_ext:
#             v = v.tolist()
#             v_ext = superadditive_extension(n, v)
#             v = torch.tensor(v_ext).to(self.device)

#         P = get_associated_game_matrix_P(coalitions, n, adj)
#         P = P.to(self.device)
#         H = torch.sparse.mm(P, torch.sparse.mm(M, P))
#         H_tilde = get_limit_game_matrix(H, is_sparse=True)
#         v_tilde = torch.sparse.mm(H_tilde, v.view(-1, 1)).view(-1)
    
#         scores = v_tilde[:n].cpu()
#         return scores


# # In[31]:


# preds = []
# for data in test_dataset:
#     try:
#         data.to(device)
#         data.x = data.x.float()
#         pred = model(data.x.float(), data.edge_index).softmax(-1)
#         preds += [pred]
#     except: pass
# preds = torch.concat(preds)
# payoff_avg = preds.mean(0).tolist()
# gstarx = GStarX(model, device, payoff_avg=payoff_avg)
# gstarx_explainer_results = []

# for data in tqdm(test_dataset):
#     try:
#         data = data.to(device)
#         data.x = data.x.float()
#         r = {
#             'data': data,
#             'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
#             'res':{}
#         }
#         soft_mask = torch.tensor(gstarx.explain(data, superadditive_ext=False, num_samples=5))
#         r['soft_mask'] = soft_mask
#         hard_masks = generate_hard_masks(soft_mask)
#         for sparsity, hard_mask in hard_masks:
#             # print(sparsity)
#             r['res'][sparsity.item()] = calculate_fidelity(data, hard_mask, model)
#             r['res'][sparsity.item()]['hard_mask'] = hard_mask
#         r['res_topk'] = {
#             1: calculate_fidelity_topk(data, soft_mask, model,1),
#             3: calculate_fidelity_topk(data, soft_mask, model,3),
#             5: calculate_fidelity_topk(data, soft_mask, model,5)
#         }
#         gstarx_explainer_results.append(r)
#     except:
#         pass


# # In[32]:


# with open(f'post_hoc/{dataset_name}/{seed}/gstarx.pkl', 'wb') as f:
#     pickle.dump(gstarx_explainer_results, f)
# gstarx_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/gstarx.pkl', 'rb'))


# # # SubGraphX

# In[34]:


# from graphxai.explainers import SubgraphX
# subgraphx_explainer = SubgraphX(model, sample_num=10)

# subgraphx_explainer_results = []
# for data in tqdm(test_dataset):
#     try:
#         data = data.to(device)
#         data.x = data.x.float()
#         r = {
#             'data': data,
#             'pred': model(data.x.float(), data.edge_index).softmax(-1).detach().cpu().numpy(),
#             'res':{}
#         }
#         exp = subgraphx_explainer.get_explanation_graph(x=data.x.float(), edge_index=data.edge_index, label=torch.tensor(r['pred'].argmax(-1)))
#         soft_mask = exp.node_imp
#         r['soft_mask'] = soft_mask
#         hard_masks = generate_hard_masks(soft_mask)
#         for sparsity, hard_mask in hard_masks:
#             # print(sparsity)
#             r['res'][sparsity.item()] = calculate_fidelity(data, hard_mask, model)
#             r['res'][sparsity.item()]['hard_mask'] = hard_mask
#         r['res_topk'] = {
#             1: calculate_fidelity_topk(data, soft_mask, model,1),
#             3: calculate_fidelity_topk(data, soft_mask, model,3),
#             5: calculate_fidelity_topk(data, soft_mask, model,5)
#         }
#     except: 
#         continue
#     subgraphx_explainer_results.append(r)


# # In[35]:


# with open(f'post_hoc/{dataset_name}/{seed}/subgraphx.pkl', 'wb') as f:
#     pickle.dump(subgraphx_explainer_results, f)
# subgraphx_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/subgraphx.pkl', 'rb'))




# # def train_epoch(model_tell, loader, device, optimizer, num_classes, reg=1, sqrt_reg=False):
# #     model_tell.train()
    
# #     total_loss = 0
# #     total_correct = 0
    
# #     for data in loader:
# #         try:
# #             loss = 0
# #             if data.x is None:
# #                 data.x = torch.ones((data.num_nodes, model_tell.num_features))
# #             if data.y.numel() == 0: continue
# #             if data.x.isnan().any(): continue
# #             if data.y.isnan().any(): continue
# #             data.x = data.x.float()
# #             y = data.y.reshape(-1).to(device).long()
# #             optimizer.zero_grad()

# #             out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))       
# #             pred = out.argmax(-1)
# #             loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
# #             # loss += reg*(torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean() + model_tell.fc.phi_in.entropy)
# #             loss += reg*model_tell.fc.phi_in.entropy
# #             if sqrt_reg:
# #                 loss+= reg*torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean()
# #             else:
# #                 loss+=reg*model_tell.fc.reg_loss
# #             loss.backward()
# #             zero_nan_gradients(model_tell)#torch.sqrt(torch.clamp(model_tell.fc.weight, min=1e-5)).sum(-1).mean()  + 
# #             optimizer.step()
# #             total_loss += loss.item() * data.num_graphs / len(loader.dataset)
# #             total_correct += pred.eq(y).sum().item() / len(loader.dataset)
# #         except Exception as e:
# #             print(e)
# #             pass

# #     return total_loss, total_correct

# # model_tell = GIN(num_classes=num_classes, num_features=num_features, num_layers=num_layers, hidden_dim=hidden_dim, nogumbel=True)
# # model_tell.load_state_dict(torch.load(os.path.join(results_path, 'best.pt'), map_location=device))
# # model_tell = model_tell.to(device)



# # model_tell.fc = LogicalLayer(model_tell.fc1.in_features, num_classes).to(device)
# # model_tell.fc.phi_in.tau = 10

# # def forward_tell(self):
# #     def fwd(x, edge_index, batch=None, activations=False, *args, **kwargs):
# #         if batch is None:
# #             batch = torch.zeros(x.shape[0]).long().to(x.device)
# #         xs = []
# #         for conv in self.convs:
# #             x = conv(x, edge_index)
# #             xs.append(x)
# #             x = self.dropout(x)
    
# #         x_mean = global_mean_pool(torch.hstack(xs), batch)
# #         x_max = global_max_pool(torch.hstack(xs), batch)
# #         x_sum = global_add_pool(torch.hstack(xs), batch)
# #         x = torch.hstack([x_mean, x_max, x_sum])
# #         # x = self.dropout(x)
# #         acts = self.fc.phi_in(x)
# #         x = self.fc(x)
# #         if activations:
# #             return x, acts, xs
# #         return x
# #     return fwd

# # # #TOREMOVE

# # # model_tell = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/tell_model.pkl', 'rb'))


# # # #UNTIL HERE


# # model_tell.forward = forward_tell(model_tell)
# # model_tell.fc.phi_in.w.shape
# # optimizer = torch.optim.Adam(model_tell.fc.parameters(), lr=0.001, weight_decay=0)
# # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# # for i in range(2000):
# #     train_loss, train_acc = train_epoch(model_tell, train_loader, device, optimizer, num_classes, reg=0.1 if i<=800 else 0.01, sqrt_reg=i>800)
# #     val_acc = test_epoch(model_tell, val_loader, device)
# #     test_acc = test_epoch(model_tell, test_loader, device)
# #     if i%10 == 0:
# #         print(i, train_loss, train_acc, val_acc, test_acc, (model_tell.fc.weight>1e-4).sum())


# # model_tell.forward = forward_tell(model_tell)


# # from torch_geometric.utils import k_hop_subgraph
# # feat_map = []
# # for readout in ['mean', 'max', 'sum']:
# #     for l in range(num_layers):
# #         for d in range(hidden_dim):
# #             feat_map.append((readout, l, d))

# # tell_explainer_results = []
# # for data in tqdm(test_dataset):
# #     try:
# #         data = data.to(device)
# #         data.x = data.x.float()
# #         pred_tell, rule_acts, layers_acts = model_tell(data.x.float(), data.edge_index, activations=True)
        
# #         pred = model(data.x.float(), data.edge_index)
# #         # rule_acts = rule_acts>0.5
# #         r = {
# #             'data': data,
# #             'pred': pred.softmax(-1).detach().cpu().numpy(),
# #             'res':{}
# #         }
# #         pred_c = r['pred'].argmax(-1).item()
# #         rules = extract_rules(model_tell.fc)
# #         soft_mask = torch.zeros(data.x.shape[0]).to(device)
# #         for c, class_rules in enumerate(rules):
# #             for rule in class_rules:
# #                 # print(rule)
# #                 # print(rule_acts[:,rule])
                
# #                 # if not rule_activated: continue
# #                 for literal in rule:
# #                     # print(literal)
# #                     agg, layer, i = feat_map[literal]
# #                     acts = layers_acts[layer][:,i]
# #                     m = torch.zeros_like(soft_mask)
# #                     if agg == 'max':
# #                         m[acts>=acts.max()] = (1 if pred_c==c else -1)*acts.max()*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
# #                     elif agg == 'sum':
# #                         m=(1 if pred_c==c else -1)*acts*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
# #                     else:
# #                         m=(1 if pred_c==c else -1)*acts*rule_acts[:,literal].item()*model_tell.fc.weight[c,literal]
# #                     m_=torch.zeros_like(m)
# #                     for i in range(len(m)):
# #                         if m[i] > 0:
# #                             try:
# #                                 subset, _, _, _ = k_hop_subgraph(i, 1, data.edge_index.cpu())
# #                                 m_[subset] += m[i]
# #                             except:
# #                                 m_[i] = m[i]
# #                     soft_mask+=m_    

# #         # print(soft_mask)
# #         soft_mask = soft_mask.detach().cpu()
# #         r['soft_mask'] = soft_mask
# #         hard_masks = generate_hard_masks(soft_mask)
# #         for sparsity, hard_mask in hard_masks:
# #             sparsity = sparsity.item()
# #             r['res'][sparsity] = calculate_fidelity(data, hard_mask, model)
# #             r['res'][sparsity]['hard_mask'] = hard_mask
# #         r['res_topk'] = {
# #             1: calculate_fidelity_topk(data, soft_mask, model,1),
# #             3: calculate_fidelity_topk(data, soft_mask, model,3),
# #             5: calculate_fidelity_topk(data, soft_mask, model,5)
# #         }
# #         tell_explainer_results.append(r)
# #     except Exception as e:
# #         print(e)


# # model_tell.forward = None

# # with open(f'post_hoc/{dataset_name}/{seed}/tell_model.pkl', 'wb') as f:
# #     pickle.dump(model_tell, f)


# # with open(f'post_hoc/{dataset_name}/{seed}/tell.pkl', 'wb') as f:
# #     pickle.dump(tell_explainer_results, f)
# # torch.save(model_tell, f'post_hoc/{dataset_name}/{seed}/model_tell.pt')
# # tell_explainer_results = pickle.load(open(f'post_hoc/{dataset_name}/{seed}/tell.pkl', 'rb'))






