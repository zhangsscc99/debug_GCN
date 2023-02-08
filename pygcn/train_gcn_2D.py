from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import copy


import torch
import torch.nn.functional as F
import torch.optim as optim


from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
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
adj, features, labels, idx_train, idx_val, idx_test = load_data()

from pandas import *
import pandas
 
# reading CSV file
# data preparation
# node matrix-> adj matrix
def data_AU(path):
    data = read_csv(path)
    
    # converting column data to list
    AU1 = data['AU01_c'].tolist()
    #AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
    AU2=data['AU02_c'].tolist()

    AU4=data['AU04_c'].tolist()

    #
    AU5=data['AU05_c'].tolist()
    AU6=data['AU06_c'].tolist()
    AU7=data['AU07_c'].tolist()

    AU9=data['AU09_c'].tolist()


    AU10=data['AU10_c'].tolist()
    AU12=data['AU12_c'].tolist()
    AU14=data['AU14_c'].tolist()

    AU15=data['AU15_c'].tolist()
    AU17=data['AU17_c'].tolist()
    AU20=data['AU20_c'].tolist()


    AU23=data['AU23_c'].tolist()
    

    AU25=data['AU25_c'].tolist()
    AU26=data['AU26_c'].tolist()

    AU28=data['AU28_c'].tolist()
    AU45=data['AU45_c'].tolist()
    AU_lst=[AU1,AU2,AU4,AU5,AU6,AU7,AU9,AU10,AU12,AU14,AU15,AU17,AU20,AU23,AU25,AU26,AU28,AU45]


    lst=[]
    for i in range(18):
        print(int(AU_lst[i][0]))
        lst.append(int(AU_lst[i][0]))

    global AU_set_lst
    AU_set_lst=[]
    for i in range(len(AU_lst[0])):
        set=[]
        for j in range(len(AU_lst)):
            set.append(AU_lst[j][i])
        AU_set_lst.append(set)
    
  

    AU_set_lst = torch.LongTensor(AU_set_lst)
    #AU_set_lst = AU_set_lst.to(torch.float32)
    embedding = torch.nn.Embedding(num_embeddings=18, embedding_dim=40)

    #features=embedding(AU_set_lst[0])
    #len(AU_set_lst)==1050  1320
    #print(len(AU_set_lst))
    #features=embedding(AU_set_lst[0])

    features=embedding(AU_set_lst[0])
    print(features.shape)

    return features


#file paths reading
paths_features=os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind')
def add_prefix(paths):
    for i in range(len(paths)):
        paths[i]='/Users/mac/Desktop/AUGCN/pygcn/Training/csvNorthwind/'+paths[i]
    return paths
paths_features=add_prefix(paths_features)
features=data_AU(paths_features[0])

#labels=[[100]]*1  #PHQ score
#label setting
labels=[[100]]*1
labels = torch.LongTensor(labels)
labels = labels.to(torch.float32)

## adj matrix creation
def adj_matrix(AU_inc1,AU_inc2,feature):
    cnt1=0
    cnt2=0
    cnt_joint=0

    for i in range(len(feature)):
        if  feature[i][AU_inc1]==1.0:
            cnt1+=1
        if  feature[i][AU_inc2]==1.0:
            cnt2+=1
        if  feature[i][AU_inc2]==1.0 and feature[i][AU_inc1]==1.0:
            cnt_joint+=1
    

    AU1_AU2_joint_count = cnt_joint # Number of instances where both AU1 and AU4 are present
    AU2_count = cnt2 # Number of instances where AU4 is present
    total_count = len(feature) # Total number of instances in the dataset

    P_AU1_AU2 = AU1_AU2_joint_count / total_count
    P_AU2 = AU2_count / total_count
    #P_AU1_given_AU2 = P_AU1_AU2 / P_AU2
    if AU2_count==0.0:
        return 0.0
    P12=AU1_AU2_joint_count/AU2_count
    return P12

def get_all_adj():
    res=[]

    for i in range(18):
        for j in range(18):
            res.append(adj_matrix(i,j,AU_set_lst))
                   #导入numpy模块，并重命名为np
    x = np.array(res)     #x是一维数组 
    d = x.reshape((18,18))                #将x重塑为2行4列的二维数组
    return d


adj=get_all_adj()
adj=torch.FloatTensor(adj)


#3D dimensional adj matrix
def format_adj(adj):
    new=[]
    for i in range(1320):
        new.append(adj)
    return new


#adj=format_adj(adj)


#index setting
#idx_train=[0]
#idx_val=[0]
idx_train=[]
for i in range(1):
    idx_train.append(i)
idx_val=[]
for i in range(1):
    idx_val.append(i)



#features=copy.deepcopy(features_AU)

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            #nclass=labels.max().item() + 1,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

#the loss function I set. MSE
loss_func = torch.nn.MSELoss()




#model training 
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    print("hello there")
    print(labels.shape)
    print(output.shape)
    print(labels)
    print(len(labels))
    print(output)
    #print(output[idx_train].shape)
    #print(labels.shape)
    #print(labels[idx_train].shape)
    #print(output)
    #print(labels)
    #loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train=loss_func(output[idx_train], labels[idx_train])   
    
    #acc_train = accuracy(output[idx_train], labels[idx_train])

    
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val=loss_func(output[idx_val], labels[idx_val])
    #acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          #'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          #'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    print(output.shape)
    #loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    loss_test=loss_func(output[idx_train], labels[idx_train])
    #acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
train(50)
test()

print(features.shape)
#from torchsummary import summary
#print(summary(model))


