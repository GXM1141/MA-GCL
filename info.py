import torch
import torch.nn as nn
from typing import Optional
import os.path as osp
import random
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pGRACE.dataset import get_dataset
from pGRACE.eval import MulticlassEvaluator
from pGRACE.model import LogReg
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')

class LogReg(nn.Module):
    def __init__(self, in_feat, num_class):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_feat, num_class)
        for m in self.modules():
            self.weight_init(m)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias != None:
                m.bias.data.fill_(0.0)

    def forward(self, feats):
        return self.fc(feats)

def log_regression(z1,
                   z2, 
                   dataset,
                   evaluator,
                   device,
                   num_epochs: int = 5000,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
    z1 = torch.from_numpy(z1).to(device)
    z2 = torch.from_numpy(z2).to(device)
    num_hidden = z1.size(1)
    y = dataset[0].y.view(-1).to(device)
    num_classes = dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(device)
    classifier2 = LogReg(num_hidden, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    optimizer2 = Adam(classifier2.parameters(), lr=0.01, weight_decay=0.0)
    torch_seed = 0
    torch.manual_seed(torch_seed)
    random.seed(12345)
    split = get_idx_split(dataset, split, preload_split)
    split = {k: v.to(device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)

    nll_loss = nn.NLLLoss()

    best_test_acc = 0
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        classifier.train()
        classifier2.train()
        optimizer.zero_grad()
        optimizer2.zero_grad()
        out1 = classifier(z1)
        out2 = classifier(z2)
        out3 = classifier2(z1)
        out4 = classifier2(z2)
        output = classifier(z1[split['train']])
        loss = nll_loss(f(out1)[split['train']], y[split['train']]) + nll_loss(f(out2)[split['train']], y[split['train']])\
             - F.kl_div(f(out3)[split['train']], F.softmax(out4,dim=-1)[split['train']]) - F.kl_div(f(out4)[split['train']], F.softmax(out3,dim=-1)[split['train']])

        loss.backward()
        optimizer.step()
        optimizer2.step()

        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                # val split is available
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z2[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z2[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    info1 = nll_loss(f(out1), y).detach()
                    info2 = nll_loss(f(out2), y).detach()
                    info3 = F.kl_div(f(out3), F.softmax(out4,dim=-1)).detach()
                    info4 = F.kl_div(f(out4), F.softmax(out3,dim=-1)).detach()
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z2[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                if best_test_acc < acc:
                    best_test_acc = acc
                    best_epoch = epoch
                    best_loss = loss
                    info1 = nll_loss(f(out1), y).detach()
                    info2 = nll_loss(f(out2), y).detach()
                    info3 = F.kl_div(F.softmax(out3,dim=-1), F.softmax(out4,dim=-1)).detach()
                    info4 = F.kl_div(F.softmax(out4,dim=-1), F.softmax(out3,dim=-1)).detach()
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}')

    return {'acc': best_test_acc}, info1, info2, info3, info4


z1 = np.load('embedding/CiteSeerview1_embeddingnew2.npy')
z2 = np.load('embedding/CiteSeerview2_embeddingnew2.npy')
dataset = 'CiteSeer'

path = osp.expanduser('~/datasets')
path = osp.join(path, dataset)
dataset = get_dataset(path, dataset)

device = 'cuda:1'
evaluator = MulticlassEvaluator()
data = dataset[0]
data = data.to(device)
split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
res, info1, info2, info3, info4 = log_regression(z1, z2, dataset, evaluator, device, split='rand:0.1', num_epochs=3000, preload_split=split)

print (res['acc'])
print (info1)
print (info2)
print (info3)
print (info4)