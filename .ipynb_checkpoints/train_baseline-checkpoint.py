from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from utils.syn_dataset import SynGraphDataset
from utils.spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils.utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json
from models.model import GIN
from torch.optim.lr_scheduler import ReduceLROnPlateau
SEEDS = 10

def train_epoch(model, loader, device, optimizer, num_classes):
    model.train()
    total_loss = 0
    total_correct = 0
    for data in loader:
        mask = [i in data.batch for i in range(data.y.shape[0])]
        y = data.y.squeeze(-1).to(device).long()
        optimizer.zero_grad()
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        out = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1)
        pred = out.argmax(-1)
        try:
            loss = F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1))
            nll_loss = F.nll_loss(F.log_softmax(out, dim=-1), y.long())
            loss = nll_loss + loss
            loss.backward()
            zero_nan_gradients(model)
            # torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        except: continue
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        total_correct += pred.eq(y).sum().item()
    train_loss = total_loss / len(loader.dataset)
    train_acc = total_correct / len(loader.dataset)

    return train_loss, train_acc

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        y = data.y.squeeze(-1).to(device)
        pred = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1000).argmax(-1)
        total_correct += pred.eq(y).sum().item()
    val_acc = total_correct / len(loader.dataset)
    
    return val_acc

def train_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)
    shutil.rmtree(path)
    path = create_folder(dataset_name, args, seed=seed)

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)

    dataset = get_dataset(dataset_name)

    
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10
    
    indices = list(range(len(dataset)))
    train_indices, val_test_indices = train_test_split(indices, test_size=0.2,
    shuffle=True, stratify=dataset.data.y, random_state=seed)

    val_indices = val_test_indices[:len(val_test_indices)//2]
    test_indices = val_test_indices[len(val_test_indices)//2:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, f)

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], nogumbel=args['nogumbel']).to(device)
    
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, min_lr=1e-5, verbose=True)

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    max_patience = 200
    patience = 0
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, device, optimizer, num_classes)
        val_acc = test_epoch(model, val_loader, device)
        test_acc = test_epoch(model, test_loader, device)
        scheduler.step(val_acc)

        if val_acc >= best_val_acc:
            patience = 0
            torch.save(model.state_dict(), os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc
        elif epoch > args['epochs']//2: patience += 1
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if patience >= max_patience: break
    
    torch.save(model.state_dict(), os.path.join(path, 'last.pt'))
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_acc = test_epoch(model, val_loader, device)
    test_acc = test_epoch(model, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results

def eval_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)
    
    dataset = get_dataset(dataset_name)

    
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10
    
    indices = list(range(len(dataset)))
    train_indices, val_test_indices = train_test_split(indices, test_size=0.2,
    shuffle=True, stratify=dataset.data.y, random_state=seed)

    val_indices = val_test_indices[:len(val_test_indices)//2]
    test_indices = val_test_indices[len(val_test_indices)//2:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], nogumbel=args['nogumbel']).to(device)
    
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_acc = test_epoch(model, val_loader, device)
    test_acc = test_epoch(model, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def train(dataset_name, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    path = create_folder(dataset_name, args)

    results = []
    for seed in range(10):
        results.append(train_seed(dataset_name, args, seed, device))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_acc_mean': df['val_acc'].mean(),
        'test_acc_mean': df['test_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_std': df['test_acc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret


def train_eval(dataset_name,  args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    
    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    
    path = create_folder(dataset_name, args)
    print(path)
    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]
    
    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, args, seed, device)
                results.append(r)
                print(r)
            except Exception as e: print(e)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_acc_mean': df['val_acc'].mean(),
        'test_acc_mean': df['test_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_std': df['test_acc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='PROTEINS', type=str,     help='Dataset to use')
    parser.add_argument('--epochs',        default=1000,       type=int,     help='Epochs')
    parser.add_argument('--hidden_dim',    default=128,        type=int,     help='Hid Dim')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--num_layers',    default=5,          type=int,     help='Number of Convolutional Layers')
    parser.add_argument('--dropout',       default=0.15,       type=float,   help='Dropout')
    parser.add_argument('--lr',            default=0.01,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=1e-5,      type=float,   help='Weight Decay')
    parser.add_argument('--nogumbel',      action='store_true',             help='Number of Convolutional Layers')
    parser.add_argument('--only_eval',    action='store_true',              help='Number of Convolutional Layers')
    parser.add_argument('--seed',          default=None,      type=int,    help='Number of Convolutional Layers')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    train_eval(dataset_name, args)

    