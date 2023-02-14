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
from ssgc import Net
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

def gen_ran_output(data, model, vice_model):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        std = torch.where(torch.isnan(param.data.std()), torch.full_like(param.data.std(), 0), param.data.std())
        adv_param.data = param.data + 0.005 * torch.normal(0,torch.ones_like(param.data)*std).to(device)           
    z2 = vice_model(data.x, data.edge_index, [2, 2], final = False)
    return z2
def train():
    """
    view_learner.train()
    view_optimizer.zero_grad()
    model.eval()

    edge_logits = view_learner(sub_x, edge_index)
    temperature = 0.5
    bias = 0.0 + 0.0001  # If bias is 0, we run into problems
    eps = 1e-20#torch.rand(n)长度为1的均匀分布采样
    gate_inputs = -torch.log(-torch.log(torch.rand(edge_logits.size()) + eps) + eps)
    gate_inputs = gate_inputs.to(device)
    gate_inputs = (gate_inputs + edge_logits) / temperature
    aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
    y = torch.zeros_like(aug_edge_weight)
    drope = torch.where(aug_edge_weight > 0.5, aug_edge_weight, y)
    
    index = torch.nonzero(drope)
    
    aug_hard = torch.zeros_like(aug_edge_weight)
    aug_hard[index] = 1.
    aug_hard = (aug_hard - aug_edge_weight).detach() + aug_edge_weight#****
    aug_hard = aug_hard.to(torch.bool)
    aug_edge_index = edge_index[:, aug_hard]
    reg = aug_edge_weight.sum() / (edge_index.shape[1] * 1.)
    #x_2 = drop_feature(sub_x, drop_feature_rate_2)
    #x_3 = drop_feature(sub_emb, 0.4)
    z1_t = model(sub_emb, identity, False)
    z2_t = model(sub_x, edge_index, False)
    z1_s = view_learner.encoder(sub_emb, identity, False)
    z2_ss = view_learner.encoder(sub_x, aug_edge_index, False)
    z2_s = view_learner.predict(z2_ss)
    loss_1 = view_learner.predLoss(z2_s, z2_t.detach())
    loss_2 = view_learner.GAELoss(z2_ss)
    loss = loss_1 + loss_2 + 0.001 * reg
    print (index.shape[0])
    print (loss_1.item())
    #print (loss_2.item())
    print (reg.item())
    

    loss.backward()
    view_optimizer.step()
    """
    model.train()
    #view_learner.eval()
    optimizer.zero_grad()
    """
    edge_logits = view_learner(sub_x, edge_index)
    temperature = 0.5
    bias = 0.0 + 0.0001  # If bias is 0, we run into problems
    eps = 1e-20#torch.rand(n)长度为1的均匀分布采样
    gate_inputs = -torch.log(-torch.log(torch.rand(edge_logits.size()) + eps) + eps)
    gate_inputs = gate_inputs.to(device)
    gate_inputs = (gate_inputs + edge_logits) / temperature
    
    aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
    y = torch.zeros_like(aug_edge_weight)
    drope = torch.where(aug_edge_weight > 0.5, aug_edge_weight, y)
    
    index = torch.nonzero(drope)
    aug_hard = torch.zeros_like(aug_edge_weight)
    aug_hard[index] = 1.
    aug_hard = (aug_hard - aug_edge_weight).detach() + aug_edge_weight#****
    aug_hard = aug_hard.to(torch.bool)
    aug_edge_index = edge_index[:, aug_hard]
    """

    z1 = model(data.x, data.edge_index, [2,2], final = False)
    z2 = gen_ran_output(data, model, vice_model)

    loss = model.loss(z1, z2, batch_size=64 if args.dataset == 'Coauthor-Phy' or args.dataset == 'ogbn-arxiv' else None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):

    model.eval()
    z3 = model(data.x, data.edge_index, [0,0], final=True)
    z = z3
    #z = (z1 + z2) * 0.5

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
            #acc = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)['acc']
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
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='Amazon-Computers')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=1)#0, 1, 2022, 2023
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    #torch.manual_seed(args.seed)
    #random.seed(12345)
    #np.random.seed(args.seed)
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

    adj = 0
    
    #if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or  args.dataset == 'PubMed': split = (data.train_mask, data.val_mask, data.test_mask)

    encoder = NewEncoder(dataset.num_features, num_hidden, get_activation(activation),
                      base_model=NewGConv, k=num_layers).to(device)
    #view_encoder = Encoder(dataset.num_features, num_hidden, get_activation(activation),
    #                  base_model=get_base_model(base_model), k=num_layers).to(device)
    #view_learner = ViewLearner(view_encoder, TA, 64).to(device)
    #view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=0.0005)
    #encoder = Net(dataset.num_features, param['num_hidden']).to(device)

    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    vice_model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=0
    )   

    log = args.verbose.split(',')
    #col = torch.from_numpy(np.array(range(adj.shape[0]))).unsqueeze(0)
    #row = torch.from_numpy(np.array(range(adj.shape[0]))).unsqueeze(0)
    #identity = torch.cat([col, row], 0).long().to(device)

    for epoch in range(1, 100 + 1):

        loss = train()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        
        if epoch == 100:
            model.eval()
            x_1 = drop_feature(data.x, drop_feature_rate_1)#3
            x_2 = drop_feature(data.x, drop_feature_rate_2)#4
            #x_3 = drop_feature(sub_x, drop_feature_rate_1)

            edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0] #adjacency with edge droprate 2
            edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0] #adjacency with edge droprate 2
            z = model(data.x, data.edge_index, [1, 1], final = True).detach().cpu().numpy()
            z1 = model(x_1, edge_index_1, [1, 1], final = True).detach().cpu().numpy()
            z2 = model(x_2, edge_index_2, [1, 1], final = True).detach().cpu().numpy()
            np.save('embedding/'+args.dataset + 'view1_embeddingsim.npy', z1)
            np.save('embedding/'+args.dataset + 'view2_embeddingsim.npy', z2)
            np.save('embedding/'+args.dataset + 'Graph_embeddingsim.npy', z)

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
