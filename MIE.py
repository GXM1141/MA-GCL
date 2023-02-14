from typing import Optional
import os.path as osp
import random
import torch
from torch.autograd import Variable
import tqdm
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pGRACE.dataset import get_dataset
from pGRACE.eval import MulticlassEvaluator
from pGRACE.model import LogReg
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality



hidden_size = 64
n_epoch = 1500
x_size = 128

class MINE(nn.Module):
    def __init__(self, in_size, hidden_size=10):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        self.xT = nn.Linear(x_size, num_classes)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    def forward(self, x, y):
        #x = self.xT(x)
        #x = F.softmax(x, dim=1)
        batch_size = x.size(0)
        print (x.shape)
        tiled_x = torch.cat([x, x, ], dim=0)
        print (tiled_x.shape)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        print (concat_y.shape)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        print (inputs.shape)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss
device = 'cuda:1'
z1 = np.load('embedding/Amazon-Computersview1_embeddingfull.npy')
z2 = np.load('embedding/Amazon-Computersview2_embeddingfull.npy')
dataset = 'Cora'
path = osp.expanduser('~/datasets')
path = osp.join(path, dataset)
dataset = get_dataset(path, dataset)
label = dataset[0].y.view(-1)
label = label.unsqueeze(1)
num_classes = dataset[0].y.max().item() + 1
onehot = torch.zeros(label.shape[0], num_classes)
onehot.scatter_(1, label, 1)
x_sample = z1
y_sample = z2

x_sample = torch.from_numpy(x_sample).float().to(device)
#y_sample = onehot.float().to(device)
y_sample = torch.from_numpy(y_sample).float().to(device)

in_dim = y_sample.shape[1]

model = MINE(in_dim, hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
plot_loss = []
all_mi = []

dataset = 'PubMed'
for epoch in range(n_epoch):

    loss = model(x_sample, y_sample)

    model.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch > 900):
        all_mi.append(-loss.cpu().item())

print (np.mean(all_mi))
