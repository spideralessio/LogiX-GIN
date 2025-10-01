import torch
import random
import numpy as np
import os
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, OGB_MAG, MoleculeNet
from ba_multi_shapes import BAMultiShapesDataset
from syn_dataset import SynGraphDataset
from spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json

def create_folder(dataset_name, args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    path = f'results/{dataset_name}/{args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def create_folder_logic(dataset_name, args, baseline_args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    baseline_args_s = '|'.join([f"{k}={baseline_args[k]}" for k in sorted(baseline_args.keys())])
    path = f'results_logic/{dataset_name}/{args_s}/{baseline_args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def get_dataset(dataset_name):
    if dataset_name == 'Ba2Motifs':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifs')
    elif dataset_name == 'Ba2MotifsNoisy':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifsnoisy')
    elif dataset_name == 'TreeGrid':
        return  SynGraphDataset(root='data/tree_grid', name='tree_grid')
    elif dataset_name == 'BaShapes':
        return  SynGraphDataset(root='data/ba_shapes', name='ba_shapes')
    elif dataset_name == 'BaCommunity':
        return  SynGraphDataset(root='data/ba_community', name='ba_community')

    elif dataset_name == 'SPMotif':
        return SPMotif(root='data/SPMotif-0.333', mode='train', transform=None)

    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        return Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    elif dataset_name == 'OGB_MAG':
        return OGB_MAG(root=f'data/{dataset_name}')
    elif dataset_name == 'BBBP':
        return MoleculeNet(name=dataset_name, root=f'data/{dataset_name}')
    elif dataset_name == 'BaMultiShapes':
        return BAMultiShapesDataset(root=f'data/{dataset_name}')
    return TUDataset(root=f'data/{dataset_name}', name=dataset_name, use_node_attr=True)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)

def zero_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad[param.grad != param.grad] = 0  # Set NaN gradients to 0