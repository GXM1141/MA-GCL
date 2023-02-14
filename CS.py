import argparse
import os.path as osp
import random
import nni
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected
import torch.nn.functional as F
import networkx as nx
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

from torch_geometric.loader import NeighborLoader
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, ViewLearner, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
def train():
    model.train()

    total_examples = total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        batch_size = batch.batch_size

        edge_index_1 = dropout_adj(batch.edge_index, p=drop_edge_rate_1)[0].to(device)
        edge_index_2 = dropout_adj(batch.edge_index, p=drop_edge_rate_2)[0].to(device) #adjacency with edge droprate 2

        x_1 = drop_feature(batch.x, 0.3).to(device)
        x_2 = drop_feature(batch.x, 0.4).to(device)

        #if drop_scheme in ['pr', 'degree', 'evc']:
        #    x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_1)
        #    x_2 = drop_feature_weighted_2(data.x, feature_weights, drop_feature_rate_2)

        z1 = model(x_1, edge_index_1, [4, 0])
        z2 = model(x_2, edge_index_2, [0, 4])
        z3 = model(x_1, edge_index_1, [2, 2])

        loss = model.loss(z1, z2, z3, batch_size=256 if args.dataset == 'Coaut' else None)
        loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(loss) * batch_size
    return total_loss / total_examples
        
def test(final=False):

    model.eval()
    z = []

    for batch in loader:
        batch = batch.to(device, 'edge_index')
        batch = batch.to(device, 'x')
        batch_size = batch.batch_size
        z_batch = model(batch.x, batch.edge_index, [4, 0])
        z.append(z_batch)
    
    z = torch.cat(z, dim = -1)

    evaluator = MulticlassEvaluator()
    
    acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']


    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc#, acc_1, acc_2        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='Coauthor-Phy')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)#0, 1, 2022, 2023
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(12345)
    np.random.seed(args.seed)
    use_nni = args.config == 'nni'
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = config['activation']
    base_model = config['base_model']
    num_layers = config['num_layers']
    dataset = args.dataset
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_scheme = config['drop_scheme']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    #dataset = get_dataset(path, args.dataset)
    root_path = osp.expanduser('~/datasets')
    dataset = Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
    data = dataset[0]
    print (data)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
    )
    adj = data.edge_index
    TA = data.edge_index
    encoder = NewEncoder(dataset.num_features, num_hidden, get_activation(activation),
                      adj, base_model=NewGConv, k=num_layers).to(device)

    model = NewGRACE(encoder, TA, adj, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    log = args.verbose.split(',')
    for epoch in range(1, num_epochs + 1):

        loss = train()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            acc = test()

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')
                #print(f'(E) | Epoch={epoch:04d}, avg_acc1 = {acc_1}')
                #print(f'(E) | Epoch={epoch:04d}, avg_acc2 = {acc_2}')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')
        #print(f'{acc_1}')
        #print(f'{acc_2}')