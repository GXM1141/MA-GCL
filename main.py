import argparse
import os.path as osp
import random
import nni
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian
import torch.nn.functional as F
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank

def train():
    model.train()
    #view_learner.eval()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2

    x_1 = drop_feature(data.x, drop_feature_rate_1)#3
    x_2 = drop_feature(data.x, drop_feature_rate_2)#4
    #cora:3,3,6,3
    #CS:(1,2)(1,2)(2,3)(2,3)
    #AP:(3,4)(4,5)(1,2)(2,3)
    #Citseer(2,3)(3,4)(1,2)(1,2)(2,2)
    #CiteSeer(4,2)(3,2)
    #AC:(3,4)(1,4)(0,2)(1,3)
    #PubMed:(0,3)(1,3)(0,3)(0,2)
    #k2 = np.random.randint(0, 4)
    z1 = model(x_1, edge_index_1, [2, 2])
    z2 = model(x_2, edge_index_2, [8, 8])

    loss = model.loss(z1, z2, batch_size=64 if args.dataset == 'Coauthor-Phy' or args.dataset == 'ogbn-arxiv' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):

    model.eval()
    z = model(data.x, data.edge_index, [1, 1], final=True)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        accs_1 = []
        accs_2 = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed':
            #acc = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=0)['acc']
            acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=0)['acc']
        else : acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=0)['acc']
        #acc_2 = log_regression(z2, dataset, evaluator2, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final and use_nni:
        nni.report_final_result(acc)
    elif use_nni:
        nni.report_intermediate_result(acc)

    return acc#, acc_1, acc_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(0)
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
    rand_layers = config['rand_layers']

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    
    data = dataset[0]
    data = data.to(device)
    """
    adj = create_adjacency_matrix(data.edge_index)
    I = scipy.sparse.eye(adj.shape[0])
    adj, lap = normalize_adjacency_matrix(adj, I)
    adj = adj.tocoo()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.FloatTensor(adj.data.astype(np.float))).to(device)
    
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops
    edge_index = data.edge_index
    edge_weight = torch.ones((edge_index.size(1), ), 
                                     device=edge_index.device)
    fill_value = 1.
    num_nodes = data.num_nodes
    edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    edge_weight_x = edge_weight
    row, col = edge_index
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, num_nodes=num_nodes)
    deg = scatter_add(edge_weight_x, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    edge_index, edge_weight = add_self_loops(
            edge_index, -0.5*edge_weight, 1, num_nodes)
    L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
    #L_3 = L.dot(L).dot(L)
    #L_6 = L.dot(L).dot(L).dot(L).dot(L).dot(L)
    eig_fn = eigs
    eig_3 = eig_fn(L, k=num_nodes-2, which='LM', return_eigenvectors=False)
    #eig_6 = eig_fn(L_6, k=num_nodes-2, which='LM', return_eigenvectors=False)
    np.savetxt('eig_4.txt', eig_3)
    #np.savetxt('eig_6.txt', eig_6)

    adj = create_adjacency_matrix(data.edge_index)
    #idx = Rank(adj, 1000, 0.9)
    #idx = idx.squeeze(0).numpy()
    I = scipy.sparse.eye(adj.shape[0])
    TA = adj + I

    adj, lap = normalize_adjacency_matrix(adj, I)
    adj = adj.tocoo()
    TA = TA.tocoo()
    lap = lap.tocoo()
    TA = torch.sparse.LongTensor(torch.LongTensor([TA.row.tolist(), TA.col.tolist()]),
                              torch.FloatTensor(TA.data.astype(np.float)))
    #TA = TA.to_dense().float()
    adj = torch.sparse.LongTensor(torch.LongTensor([adj.row.tolist(), adj.col.tolist()]),
                              torch.FloatTensor(adj.data.astype(np.float)))
    #adj = adj.to_dense().float().to(device)
    adj = adj.float().to(device)
    #lap = torch.sparse.LongTensor(torch.LongTensor([lap.row.tolist(), lap.col.tolist()]),
   #                           torch.FloatTensor(lap.data.astype(np.float)))
    #lap = lap.to_dense().float()

    K = 8
    feat = data.x
    emb = feat
    for i in range(K):
        feat = torch.spmm(adj, feat)
        emb = emb + feat
    emb/=K
    """
    adj = 0
    
    #if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed': split = (data.train_mask, data.val_mask, data.test_mask)

    encoder = NewEncoder(dataset.num_features, num_hidden, get_activation(activation),
                      base_model=NewGConv, k=num_layers).to(device)

    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    
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
            x_1 = drop_feature(data.x, drop_feature_rate_1)#3
            x_2 = drop_feature(data.x, drop_feature_rate_2)#4
            #x_3 = drop_feature(sub_x, drop_feature_rate_1)

            edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2
            edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0] #adjacency with edge droprate 2
            z = model(data.x, data.edge_index, [2, 2], final=True).detach().cpu().numpy()
            z1 = model(x_1, edge_index_1, [2, 2], final=True).detach().cpu().numpy()
            z2 = model(x_2, edge_index_2, [2, 2], final=True).detach().cpu().numpy()
            np.save('embedding/'+args.dataset + 'view1_embeddingfull.npy', z1)
            np.save('embedding/'+args.dataset + 'view2_embeddingfull.npy', z2)
            np.save('embedding/'+args.dataset + 'Graph_embeddingfull.npy', z)
            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')


    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')

