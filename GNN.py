from mimetypes import init
from turtle import forward
from pip import main
import torch
import torch_geometric
# from torch_geometric.nn import ChebConv, GCNConv  
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.graphproppred.mol_encoder import BondEncoder, AtomEncoder

class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNN_node(torch.nn.Module):
    def __init__(self, emb_dim):
        super(GNN_node, self).__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.convs = torch.nn.ModuleList([GCNConv(emb_dim), GCNConv(emb_dim)])
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        h = self.convs[0](self.atom_encoder(x), edge_index, edge_attr)

        h = self.convs[1](h, edge_index, edge_attr)
        return h

class Net(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gnn = GNN_node(emb_dim)
        self.pool = torch_geometric.nn.global_max_pool
        self.graph_lin = torch.nn.Linear(emb_dim, 2)
        # self.conv1 = GCNConv(emb_dim)
        # self.conv2 = GCNConv(emb_dim)

        # self.atom_encoder = AtomEncoder(emb_dim=emb_dim)
        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        h_node = self.gnn(data)
        h_graph = self.pool(h_node, data.batch)
        return self.graph_lin(h_graph)
        '''
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # x = x.to(dtype=torch.float)

        x = self.atom_encoder(x)
        # edge_index = edge_index.to(torch.double)
        # edge_weight = edge_weight.to(dtype=torch.double)
        edge_embedding = self.bond_encoder(edge_weight)

        # print(x.shape, edge_index.shape, edge_weight.shape)
        print(x.dtype, edge_index.dtype, edge_embedding.dtype)
        x = self.conv1(x, edge_index, edge_embedding)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_embedding)
        return F.log_softmax(x, dim=1)
        '''

if __name__ == 'main':
    pass