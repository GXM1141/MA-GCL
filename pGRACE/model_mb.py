from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from ssgc import Net

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import TUDataset

import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

def glorot(tensor):#inits.py中
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)#将tensor的值设置为-stdv, stdv之间
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)



class NewGConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(NewGConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels#输入通道数，也就是X的shape[1]
        self.out_channels = out_channels#输出通道数
        self.improved = improved#$设置为true时A尖等于A+2I
      
        self.cached = cached#If set to True, the layer will cache the computation of D^−1/2A^D^−1/2 on first execution, and will use the cached version for further executions. This parameter should only be set to True in transductive learning scenarios. (default: False)
        self.normalize = normalize#是否添加自环并应用对称归一化。

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:#如果设置为False，则该层将不会学习加法偏差
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)#glorot函数下面有写，初始化weight矩阵
        zeros(self.bias)#zeros函数下面有写，初始化偏置矩阵
        self.cached_result = None
        self.cached_num_edges = None


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, c, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)#将x与权重矩阵相乘
    
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        for _ in range(c):
            x = x + self.propagate(edge_index, x=x, norm=norm)
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class NewEncoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GATConv, k: int = 2, skip=False):
        super(NewEncoder, self).__init__()
        self.base_model = base_model
        assert k >= 1
        self.k = k
        self.skip = skip
        self.out = out_channels
        hi = 2
        if k == 1:
            self.conv = [base_model(in_channels, out_channels).jittable()]
            self.conv = nn.ModuleList(self.conv)
            self.activation = activation
        elif not self.skip:
            self.conv = [base_model(in_channels, hi * out_channels)]
            for _ in range(1, k - 1):
                self.conv.append(base_model(hi * out_channels, hi * out_channels))
            self.conv.append(base_model(hi * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, l = [1, 1]):

        for i in range(self.k):
            K = np.random.randint(0, 4)
            feat = x
            #emb = feat
            #print (K)
            x = self.activation(self.conv[i](feat, edge_index, K))
        
        return x

class NewGRACE(torch.nn.Module):
    def __init__(self, encoder: NewEncoder, A: torch.Tensor, adj:torch.Tensor, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(NewGRACE, self).__init__()
        self.encoder: NewEncoder = encoder
        #self.encoder2: Encoder = encoder2
        self.BCE = torch.nn.BCELoss()
        self.tau: float = tau
        self.adj = adj
        self.A = A
        self.norm = (A.shape[0] * A.shape[0]) / (float((A.shape[0] * A.shape[0] - torch.sum(A))) * 2)
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, l = [0, 0]) -> torch.Tensor:
        return self.encoder(x, edge_index, l)#, self.encoder2(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        #z = self.fc1(z)
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def recLoss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        ret = -torch.log(between_sim.diag() / between_sim.sum(1))
        ret = ret.mean()
        return ret
        
    def GAELoss(self, z: torch.Tensor):
        act = nn.Sigmoid()
        return self.norm * self.BCE(act(torch.mm(z, z.t())), self.A)


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, k = 0, r=0.3):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        sim_gate = torch.sigmoid(between_sim)

        sim_f = between_sim
        pos_sim = between_sim
        for i in range(k):
            between_sim = torch.mm(self.adj, between_sim)
            pos_sim = (1 - r**(i+1)) * pos_sim + (r**(i+1)) * between_sim
        between_sim = pos_sim

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + sim_f.sum(1) - refl_sim.diag()))
    
    def bn_loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z3))
        refl2_sim = f(self.sim(z2, z3))
        between_sim = f(self.sim(z1, z2))

        #return -torch.log(refl_sim.diag() / refl_sim.sum(1)) - torch.log(refl2_sim.diag() / (refl2_sim.sum(1)))
        return -torch.log(refl_sim.diag() / (between_sim.sum(1) + refl_sim.sum(1))) - torch.log(refl2_sim.diag() / (between_sim.sum(1) + refl2_sim.sum(1)))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h3 = self.projection(z3)
        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        #ret = l1
        ret = ret.mean() if mean else ret.sum()
        #ret = ret# + 0.1 * self.GAELoss(z1) + 0.1 * self.GAELoss(z2)
        return ret


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class ViewLearner(torch.nn.Module):
    def __init__(self, encoder: Encoder, A: torch.Tensor, mlp_edge_model_dim: int = 64):
        super(ViewLearner, self).__init__()
        self.encoder: Encoder = encoder
        self.input_dim = self.encoder.out
        self.hidden = 128
        self.tau = 0.4
        self.BCE = torch.nn.BCELoss()
        self.A = A
        self.norm = (A.shape[0] * A.shape[0]) / (float((A.shape[0] * A.shape[0] - torch.sum(A))) * 2)
        self.mlp_edge_model = Sequential(
            Linear(self.input_dim * 2, mlp_edge_model_dim),
            ReLU(),
            Linear(mlp_edge_model_dim, 1)
        )
        self.predict = nn.Sequential(nn.Linear(self.input_dim, self.hidden), nn.PReLU(), nn.Linear(self.hidden, self.input_dim))
        self.init_emb()

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def predLoss(self, z1: torch.Tensor, z2: torch.Tensor):
        x = F.normalize(z1, dim=-1, p=2)
        y = F.normalize(z2, dim=-1, p=2)
        ret = 2 - 2 * (x * y).sum(dim=-1)
        ret = ret.mean()
        return ret
    def recLoss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        ret = -torch.log(between_sim.diag() / between_sim.sum(1))
        ret = ret.mean()
        return ret
        
    def GAELoss(self, z: torch.Tensor):
        act = nn.Sigmoid()
        return self.norm * self.BCE(act(torch.mm(z, z.t())), self.A)

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        node_emb = self.encoder(x, edge_index)
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logits = self.mlp_edge_model(edge_emb)

        return edge_logits
