import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from graph_conv_hetero_2 import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = True, JK = "sum", graph_pooling = "mean",
                    
                    aggregators = ['mean', 'min', 'max', 'std'], # For PNA
                    scalers = ['identity', 'amplification', 'attenuation'], # For PNA
                    deg = None, # For PNA
                    edge_dim = None, # For PNA

                    use_signnet = False, # For SignNet position encoding
                    node_dim = None, # For SignNet position encoding
                    cfg_posenc = None, # For SignNet position encoding
                
                    num_nodes = None, # Number of nodes

                    device = 'cuda'
                ):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_nodes = num_nodes

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, residual = residual, gnn_type = gnn_type, 
                    aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim, 
                    use_signnet = use_signnet, node_dim = node_dim, cfg_posenc = cfg_posenc,
                    num_nodes = num_nodes,
                    device = device).to(device = device)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, residual = residual, gnn_type = gnn_type,
                    aggregators = aggregators, scalers = scalers, deg = deg, edge_dim = edge_dim, 
                    use_signnet = use_signnet, node_dim = node_dim, cfg_posenc = cfg_posenc,
                    num_nodes = num_nodes,
                    device = device).to(device = device)
        
        # Final node-level prediction
        self.fc1 = torch.nn.Linear(emb_dim, 512)
        self.fc2 = torch.nn.Linear(512, num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        predict = self.fc2(torch.nn.functional.leaky_relu(self.fc1(h_node), negative_slope = 0.1))
        return predict

