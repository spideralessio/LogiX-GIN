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
import pandas as pd
import argparse
import pickle
import json
from model_node import GIN, GINTELL


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

def train_epoch(model, model_tell, data, mask, device, optimizer, num_classes, train_full=True, conv_reg=0.001, fc_reg=0.01):
    model.train()
    model_tell.train()
    data = data.to(device)
    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = [0]*(len(model_tell.convs)+1)
    
    y = data.y.to(device)
    
    optimizer.zero_grad()
    with torch.no_grad():
        layers_x, layers_y = model.forward_e(data.x.float(), data.edge_index, tau=1000)

    
    loss = 0
    last_layer_out = None
    for i, (layer, layer_x, layer_y) in enumerate(zip(model_tell.convs, layers_x[:-1], layers_y[:-1])):
        if train_full: layer.nn[0].phi_in.tau = 10
        layer_x = layer_x if i!=0 else model_tell.input_bnorm(layer_x)
        if i !=0: layer_x = model_tell.dropout(layer_x)
        layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
        layer_loss = F.binary_cross_entropy(layer_out[mask, :].reshape(-1), layer_y[mask, :].reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
        loss += layer_loss
        if train_full and i!=0:
            layer_x = model_tell.input_bnorm(last_layer_out)
            layer_out = layer(torch.hstack([layer_x, 1-layer_x]), data.edge_index.to(device))
            layer_loss = F.binary_cross_entropy(layer_out[mask, :].reshape(-1), layer_y[mask, :].reshape(-1)) + conv_reg* (layer.nn[0].reg_loss + layer.nn[0].phi_in.entropy)
            loss += layer_loss

        last_layer_out = layer_out
        total_loss[i] += layer_loss.item() / mask.sum().item()
        total_correct[i] += ((layer_out[mask, :] >= 0.5).long() == layer_y[mask, :]).sum().item() / (layer_y.shape[-1]*mask.sum().item())
    # out = model_tell.fc(model_tell.output_bnorm(layers_x[-1]))       
    out = model_tell.fc(torch.hstack([model_tell.output_bnorm(layers_x[-1]), 1-model_tell.output_bnorm(layers_x[-1])]))
    layer_loss = 0
    pred = out.argmax(-1)
    layer_loss += F.binary_cross_entropy(out[mask, :].reshape(-1), torch.nn.functional.one_hot(y[mask], num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out[mask, :], dim=-1), y[mask].long())
    
    if train_full:
        model_tell.fc.phi_in.tau = 10
        out = model_tell(data.x.float(), data.edge_index)       
        pred = out.argmax(-1)
        layer_loss += F.binary_cross_entropy(out[mask, :].reshape(-1), torch.nn.functional.one_hot(y[mask], num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out[mask, :], dim=-1), y[mask].long())
       
    layer_loss += fc_reg*(model_tell.fc.reg_loss + model_tell.fc.phi_in.entropy)
    loss += layer_loss
    loss.backward()
    zero_nan_gradients(model_tell)
    optimizer.step()
    total_loss[-1] += layer_loss.item() / mask.sum().item()
    total_correct[-1] += pred[mask].eq(y[mask]).sum().item() / mask.sum().item()


    return total_loss, total_correct

@torch.no_grad()
def test_epoch(model, data, mask, device):
    model.eval()
    total_correct = 0
    data = data.to(device)
    y = data.y.reshape(-1)
    pred = model(data.x.float(), data.edge_index, tau=1000).argmax(-1)
    total_correct += pred[mask].eq(y[mask]).sum().item()
    val_acc = total_correct / mask.sum().item()
    
    return val_acc
from torch_geometric.loader import NeighborLoader
def train_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

        
    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)
    shutil.rmtree(path)
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)

    os.mkdir(os.path.join(path, 'code'))
    for f in glob.glob('*.py'):
        shutil.copy(f, os.path.join(path, 'code'))

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

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'dataset': dataset_name,
        }, f)
    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    print(model)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
        
    print('Baseline Acc:', test_epoch(model, dataset.data, dataset.data.test_mask, device))
    model_tell = GINTELL(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers']).to(device)
    
    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    train_loader = NeighborLoader(
        dataset.data,
        num_neighbors=[25]*baseline_args['num_layers'],
        batch_size=args['batch_size'],
        input_nodes=dataset.data.train_mask,
    )
    for epoch in range(args['epochs']):
        for batch in train_loader:
            train_loss, train_acc = train_epoch(model, model_tell, batch, batch.train_mask, device, optimizer, num_classes, train_full=epoch>args['warmup_epochs'], conv_reg=args['conv_reg'], fc_reg=args['fc_reg'])
        val_acc = test_epoch(model_tell, dataset.data, dataset.data.val_mask, device)
        test_acc = test_epoch(model_tell, dataset.data, dataset.data.test_mask, device)

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
    # model_tell.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, dataset.data, dataset.data.val_mask, device)
    test_acc = test_epoch(model_tell, dataset.data, dataset.data.test_mask, device)

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
    
    model = GIN(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, dataset.data, dataset.data.train_mask, device))
    model_tell = GINTELL(num_features=num_features, num_classes=num_classes, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers']).to(device)
    # model_tell.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    model_tell = torch.load(os.path.join(path, 'best.pt'))
    # model_tell.load_state_dict(torch.load('best.pt'))

    val_acc = test_epoch(model_tell, dataset.data, dataset.data.val_mask, device)
    test_acc = test_epoch(model_tell, dataset.data, dataset.data.test_mask, device)

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

    seeds = range(5)
    if seed_todo is not None:
        seeds = [seed_todo]
    
    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(10):
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

    