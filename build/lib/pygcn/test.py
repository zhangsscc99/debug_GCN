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
#adj, features, labels, idx_train, idx_val, idx_test = load_data()




#1,1,1,1,1,1,
#[[1.0],
# [1.0],
# ]

#concat 
#[[]
# ]

 
# reading CSV file
# data preparation
# node matrix-> adj matrix
def data_AU(path):
    data = pd.read_csv(path)
    
    # converting column data to list
    #AU1, AU2, AU4）mouth  AU10, AU12, AU14, AU15, AU17, and AU25
    AU1 = data['AU01_c'].tolist()
    AU2 = data['AU02_c'].tolist()
    AU4 = data['AU04_c'].tolist()
    AU5 = data['AU05_c'].tolist()
    AU6 = data['AU06_c'].tolist()
    AU7 = data['AU07_c'].tolist()
    AU9 = data['AU09_c'].tolist()
    AU10 = data['AU10_c'].tolist()
    AU12 = data['AU12_c'].tolist()
    AU14 = data['AU14_c'].tolist()
    AU15 = data['AU15_c'].tolist()
    AU17 = data['AU17_c'].tolist()
    AU20 = data['AU20_c'].tolist()


    AU23=data['AU23_c'].tolist()
    

    AU25=data['AU25_c'].tolist()
    AU26=data['AU26_c'].tolist()

    AU28=data['AU28_c'].tolist()
    AU45=data['AU45_c'].tolist()
    AU_lst=[AU1,AU2,AU4,AU5,AU6,AU7,AU9,AU10,AU12,AU14,AU15,AU17,AU20,AU23,AU25,AU26,AU28,AU45]


    lst = []
    for i in range(18):
        print(int(AU_lst[i][0]))
        lst.append(int(AU_lst[i][0]))

    global AU_set_lst
    AU_set_lst = []
    for i in range(len(AU_lst[0])):
        set=[]
        for j in range(len(AU_lst)):
            set.append(AU_lst[j][i])
        AU_set_lst.append(set)
    
    AU_set_lst2 = []
    for j in range(len(AU_set_lst[0])):
        ave_AU=0
        for i in range(len(AU_set_lst)):
            ave_AU+=AU_set_lst[i][j]
        ave_AU=ave_AU/len(AU_set_lst)
        AU_set_lst2.append(ave_AU)
    AU_set_lst2 = torch.LongTensor(AU_set_lst2)
  

    AU_set_lst = torch.LongTensor(AU_set_lst)
    #AU_set_lst = AU_set_lst.to(torch.float32)

    embedding = torch.nn.Embedding(num_embeddings=18, embedding_dim=40)
    



    
    #features=embedding(AU_set_lst[0])
    #len(AU_set_lst)==1050  1320
    #print(len(AU_set_lst))
    #features=embedding(AU_set_lst[0])

    features=embedding(AU_set_lst2)
    #print(features.shape)

    return features


#file paths reading
paths_features = os.listdir('/Users/mac/Desktop/AUGCN/pygcn/Training2/csvNorthwind')
def add_prefix(paths):
    for i in range(len(paths)):
        paths[i] = '/Users/mac/Desktop/AUGCN/pygcn/Training2/csvNorthwind/'+paths[i]
    return paths
paths_features = add_prefix(paths_features)
print(len(paths_features))
print("here")
features = data_AU(paths_features[0])

#new concatenated one
for i in range(len(paths_features)):
    big_lst = []
    big_lst.append(data_AU(paths_features[0]))
features = torch.stack(big_lst,dim=0)

#labels=[[100]]*1  #PHQ score
#label setting
labels = [[90]]*45
labels = torch.LongTensor(labels)
labels = labels.to(torch.float32)

labels = []
for i in range(len(paths_features)):
    label = [100]
    label = torch.LongTensor(label)
    label = label.to(torch.float32)
    labels.append(label)
labels = torch.stack(labels,dim=0)

    

# adj matrix creation
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


adj= get_all_adj()
adj= torch.FloatTensor(adj)


#3D dimensional adj matrix
def format_adj(adj):
    adj_lst = []
    for i in range(len(paths_features)):
        adj = get_all_adj()
        adj = torch.FloatTensor(adj)
        adj_lst.append(adj)
    return torch.stack(adj_lst,dim=0)


adj = format_adj(adj)

num=adj.shape[0]
features=features.reshape(num,-1)
adj=adj.reshape(num,-1)


#index setting
#idx_train=[0]
#idx_val=[0]
idx_train = []
for i in range(1):
    idx_train.append(i)
idx_val=[]
for i in range(1):
    idx_val.append(i)

print(features)
print(features.shape[1])

#features=copy.deepcopy(features_AU)

# Model and optimizer
#model = GCN(nfeat=features.shape[1],
#            nhid=args.hidden,
#            #nclass=labels.max().item() + 1,
#            nclass=1,
#            dropout=args.dropout)
#optimizer = optim.Adam(model.parameters(),
#                       lr=args.lr, weight_decay=args.weight_decay)

#the loss function I set. MSE
#loss_func = torch.nn.MSELoss()
