import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import random
import os

def create_adjacency_matrix(edge_index):
    """
    Creating a sparse adjacency matrix.
    :param graph: NetworkX object.
    :return A: Adjacency matrix.
    """
    edges = edge_index.t().cpu().numpy()
    index_1 = [edge[0] for edge in edges] + [edge[1] for edge in edges]
    index_2 = [edge[1] for edge in edges] + [edge[0] for edge in edges]
    values = [1 for edge in index_1]
    node_count = max(max(index_1)+1, max(index_2)+1)
    A = sparse.coo_matrix((np.array(values)/2, (np.array(index_1), np.array(index_2))), shape=(node_count, node_count), dtype=np.float32)
    return A

def Rank(A, len = 1000, a = 0.8):
    I = sparse.eye(A.shape[0])
    A_s = normalize_adjacency_matrix(A, I)[0].tocoo()
    A_s = normalize_adj(A).tocoo()
    Pi = (torch.ones(A.shape[0], 1)/A.shape[0])
    A = torch.sparse.LongTensor(torch.LongTensor([A_s.row.tolist(), A_s.col.tolist()]),
                              torch.FloatTensor(A_s.data.astype(np.float)))
    
    A = A.to_dense().float()
    A_s = A
    e = (torch.ones_like(A) / A.shape[0])
    A = a * A + (1-a) * e
    for i in range (100):
        Pi = torch.mm(A, Pi)
    val, idx = torch.topk(Pi, len, dim = 0)
    idx = idx.t()
    val = val.t()
    idx, _ = torch.sort(idx)
    return idx

def normalize_adj(adj):
   adj = sparse.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

def normalize_adjacency_matrix(A, I):
   
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    A_lap = D.dot(A).dot(D)
    return A_tilde_hat, A_lap

def load_adj_neg(num_nodes, sample):
    col = np.random.randint(0, num_nodes, size=num_nodes * sample)
    row = np.repeat(range(num_nodes), sample)
    index = np.not_equal(col,row)
    col = col[index]
    row = row[index]
    new_col = np.concatenate((col,row),axis=0)
    new_row = np.concatenate((row,col),axis=0)
    #data = np.ones(num_nodes * sample*2)
    data = np.ones(new_col.shape[0])
    adj_neg = sparse.coo_matrix((data, (new_row, new_col)), shape=(num_nodes, num_nodes))
    #adj_neg = (sp.eye(adj_neg.shape[0]) * sample - adj_neg).toarray()
    #adj_neg = (sp.eye(adj_neg.shape[0]) - adj_neg/sample).toarray()
    #adj_neg = (adj_neg / sample).toarray()
    adj_neg = normalize_adj(adj_neg)

    return adj_neg.toarray()

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
