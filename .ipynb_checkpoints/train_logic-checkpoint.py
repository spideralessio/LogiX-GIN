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
from models.model import GIN, GINTELL
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEEDS = 10

def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=False')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname

def train_epoch(model, model_tell, loader, device, optimizer, num_classes, train_full=True, conv_reg=0.001, fc_reg=0.01):
    model.train()
    model_tell.train()
    
    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = [0]*(len(model_tell.convs)+1)
    
    for data in loader:
        try:
            if data.x is None:
                data.x = torch.ones((data.num_nodes, model.num_features))
            if data.y.numel() == 0: continue
            if data.x.isnan().any(): continue
            if data.y.isnan().any(): continue
            y = data.y.reshape(-1).to(device).long()
            optimizer.zero_grad()
            with torch.no_grad():
                layers_x, layers_y = model.forward_e(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))
            loss = 0
            last_layer_out = None
            for i, (layer, layer_x, layer_y) in enumerate(zip(model_tell.convs, layers_x[:-1], layers_y[:-1])):
                if train_full: layer.nn[0].phi_in.tau = 10
                if i !=0: layer_x = F.dropout(layer_x, p=0.2)*(1-0.2)
                layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
                layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
                loss += layer_loss
                if train_full and i!=0:
                    layer_x = last_layer_out
                    layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
                    layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
                    loss += layer_loss
                last_layer_out = layer_out
                total_loss[i] += layer_loss.item() / len(loader.dataset)
                total_correct[i] += ((layer_out >= 0.5).long() == layer_y).sum().item() / (layer_y.shape[-2]*layer_y.shape[-1]*len(loader))
            
            out = model_tell.fc(torch.hstack([layers_x[-1], 1-layers_x[-1]]))       
            pred = out.argmax(-1)
            loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
            
            if train_full:
                model_tell.fc.phi_in.tau = 10
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device))       
                pred = out.argmax(-1)
                loss += F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
               
            loss += fc_reg*(model_tell.fc.reg_loss + model_tell.fc.phi_in.entropy)
            
            loss.backward()
            zero_nan_gradients(model_tell)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss[-1] += loss.item() * data.num_graphs / len(loader.dataset)
            total_correct[-1] += pred.eq(y).sum().item() / len(loader.dataset)
        except Exception as e:
            print(e)
            pass

    return total_loss, total_correct

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        if data.y.numel() == 0: continue
        if data.x.isnan().any(): continue
        if data.y.isnan().any(): continue
        y = data.y.reshape(-1).to(device)
        pred = model(data.x.float().to(device), data.edge_index.to(device), data.batch.to(device), tau=1000).argmax(-1)
        total_correct += pred.eq(y).sum().item()
    val_acc = total_correct / len(loader.dataset)
    
    return val_acc

def train_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

        
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)
    shutil.rmtree(path)
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)


    dataset = get_dataset(dataset_name)

    print(f'Training logic model on {dataset_name}')
    print(baseline_args)

    
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10
    
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))

    train_dataset = dataset[data['train_indices']]
    val_dataset = dataset[data['val_indices']]
    test_dataset = dataset[data['test_indices']]
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_indices': data['train_indices'],
            'val_indices': data['val_indices'],
            'test_indices': data['test_indices'],
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, f)
    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_loader, device))
    model_tell = GINTELL(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers']).to(device)
    
    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model, model_tell, train_loader, device, optimizer, num_classes, train_full=epoch>args['warmup_epochs'], conv_reg=args['conv_reg'], fc_reg=args['fc_reg'])
        val_acc = test_epoch(model_tell, val_loader, device)
        test_acc = test_epoch(model_tell, test_loader, device)

        if epoch > args['warmup_epochs']:
            scheduler.step(val_acc)
        if epoch>args['warmup_epochs'] and val_acc >= best_val_acc:
            torch.save(model_tell, os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    
    torch.save(model_tell, os.path.join(path, 'last.pt'))
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results

def eval_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)

    dataset = get_dataset(dataset_name)

    print(f'Evaluating logic model on {dataset_name}')
    print(baseline_args)

    num_classes = dataset.num_classes
    num_features = dataset.num_features

    if num_features == 0: num_features = 10
    
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))

    train_dataset = dataset[data['train_indices']]
    val_dataset = dataset[data['val_indices']]
    test_dataset = dataset[data['test_indices']]
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_loader, device))
    model_tell = GINTELL(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers']).to(device)
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def train_eval(dataset_name, baseline_path, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))
    
    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    
    path = create_folder_logic(dataset_name, args, baseline_args)

    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]
    
    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device)
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
    parser.add_argument('--baseline_path', default=None,       type=str,     help='Baseline path')
    parser.add_argument('--epochs',        default=5000,       type=int,     help='Epochs')
    parser.add_argument('--warmup_epochs', default=3000,       type=int,     help='Epochs')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--lr',            default=0.001,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.0,      type=float,     help='Weight decay')
    parser.add_argument('--conv_reg',      default=0.001,      type=float,   help='Conv layer regularization')
    parser.add_argument('--fc_reg',        default=0.01,      type=float,    help='Last layer regularization')
    parser.add_argument('--only_eval',    action='store_true',              help='Number of Convolutional Layers')
    parser.add_argument('--seed',          default=None,      type=int,    help='Number of Convolutional Layers')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    baseline_path = args.pop('baseline_path')
    if baseline_path is None:
        baseline_path = get_best_baseline_path(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)

    