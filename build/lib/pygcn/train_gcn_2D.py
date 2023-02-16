from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from pygcn.utils import load_data
from pygcn.models import GCN
from pygcn.feature_matrix import *
from pygcn.adj_matrix import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
# reading CSV file
# data preparation
# node matrix-> adj matrix
# file paths reading
paths_features = os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind')
print(len(paths_features))
def add_prefix(paths):
    paths = sorted(paths)
    # print(paths)
    # paths.pop(0)
    for i in range(len(paths)):
        paths[i] = '/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind/'+paths[i]
    return paths

paths_features = add_prefix(paths_features)
# paths_features = sorted()
# print(paths_features)
print(paths_features)

# new concatenated one
big_lst = []
for i in range(len(paths_features)):
    try:
        features_inside = data_AU(paths_features[i])[0]
        
        big_lst.append(features_inside)
    except KeyError:
        print(str(i)+"th file has problems")

    # print(features_inside)
# print(big_lst)
# print(big_lst)
# print(type(big_lst))
features = torch.stack(big_lst, dim=0)

# print(features)

# labels=[[100]]*1  #PHQ score
# label setting
# print(len(os.listdir('/Users/mac/Desktop/AUGCN/pygcn/labels/AVEC2014_DepressionLabels/Training_DepressionLabels')))


labels = []
for i in range(len(paths_features)):
    label = [100]
    label = torch.LongTensor(label)
    label = label.to(torch.float32)
    labels.append(label)

labels = torch.stack(labels, dim=0)
# print(labels)



def get_all_adj(AU_set_lst):
    res=[]
    for i in range(18):
        for j in range(18):
            res.append(adj_matrix(i,j,AU_set_lst))
                   
    x = np.array(res)     #x是一维数组 
    d = x.reshape((18,18))                #将x重塑为2行4列的二维数组
    return d


# adj= get_all_adj()
# adj= torch.FloatTensor(adj)


# 3D dimensional adj matrix
def format_adj():
    adj_lst = []
    for i in range(len(paths_features)):
        try:
            AU_set_lst = data_AU(paths_features[i])[-1]
        
            
            adj = get_all_adj(AU_set_lst)
            adj = torch.FloatTensor(adj)
            adj_lst.append(adj)
        except KeyError:
            pass
    return torch.stack(adj_lst,dim=0)


adj = format_adj()

print(adj.shape)
print(features.shape)


# Model and optimizer
model = GCN(nfeat=features.shape[-1],
            nhid=args.hidden, #this is really a big problem
            #nclass=labels.max().item() + 1,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# the loss function I set. MSE
loss_func = torch.nn.MSELoss()

# model training 
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    # print(output)
    print(output.shape)
    # print(output[idx_train],labels[idx_train])

    idx_train = []
    for i in range(len(output)):
        idx_train.append(i)
    idx_val = []
    for i in range(len(output)):
        idx_val.append(i)
    
    
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train = loss_func(output[idx_train], labels[idx_train])   
    # acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        # print(output)


    # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val = loss_func(output[idx_val], labels[idx_val])
    # acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          # 'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          # 'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    # print(output.shape)
    idx_train = []
    for i in range(len(output)):
        idx_train.append(i)
    idx_val = []
    for i in range(len(output)):
        idx_val.append(i)
    # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test=loss_func(output[idx_train], labels[idx_train])
    # acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))


train(args.epochs)
test()

# print(features.shape)
# from torchsummary import summary
# print(summary(model))


