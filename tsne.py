from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

x = np.load('embedding/CoraGraph_embeddingariel.npy')
dataset = 'Cora'

path = osp.expanduser('~/datasets')
path = osp.join(path, dataset)
data = get_dataset(path, dataset)
label = data[0].y.view(-1)
#x = data[0].x

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(x)


print(tsne_results.shape)

# Create the figure
fig = plt.figure( figsize=(8,8) )
ax = fig.add_subplot(1, 1, 1, title='TSNE' )
col = []
colors = ['red', 'darkorange', 'darkgreen', 'dodgerblue', 'midnightblue', 'gold', 'black', 'darkolivegreen', 'purple', 'saddlebrown']

for i in range(0, len(label)):
    col.append(colors[label[i]])

# Create the scatter
ax.scatter(
    x=tsne_results[:,0],
    y=tsne_results[:,1],
    c=col,
    cmap=plt.cm.get_cmap('Paired'),
    alpha=0.4,
    s=0.5)
plt.savefig('figure/'+ dataset + '_ariel.jpg')
plt.show()