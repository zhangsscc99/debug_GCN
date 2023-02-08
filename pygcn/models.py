import torch.nn as nn
import torch.nn.functional as F
import torch
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_channels, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(in_channels * nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.linear = nn.Linear(nclass, 1)
        self.dropout = dropout
        
    def forward(self, x, adj):
        # Reshape the 3D input to 2D, where each row represents a node in the graph
        x = x.view(-1, x.shape[2])
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        return self.linear(x)

    

       

        #所以gcn确实是这样，只需要改一层就可以了，输出层就OK。一行行源码进行解构

"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        #self.gc2 = nn.Linear(nhid, 1)
        self.linear = nn.Linear(nclass, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=1)
        return self.linear(x)

    def forward(self, x, adj):
        out = []
        for i in range(x.shape[0]):
            x_i = F.relu(self.gc1(x[i], adj[i]))
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = self.gc2(x_i, adj[i])
            out.append(self.linear(x_i))
        print(1)
        print(out)
        return torch.cat(out, dim=0)


    def forward(self, x, adj):
        outputs = []
        for i in range(x.shape[0]):
            x_i = F.relu(self.gc1(x[i], adj[i]))
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = self.gc2(x_i, adj[i])
            outputs.append(self.linear(x_i))
        
        return torch.cat(outputs, dim=0)
"""
