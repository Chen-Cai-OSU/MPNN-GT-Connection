import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer


@register_network('custom_virtual_gnn')
class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        layers = []
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(dim_in,
                                     dim_in,
                                     dropout=cfg.gnn.dropout,
                                     residual=cfg.gnn.residual))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

        # added by Chen; VN part.
        self.virtualnode_embedding = torch.nn.Embedding(1, dim_in)
        self.mlp_virtualnode_list = torch.nn.ModuleList()
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        for layer in range(cfg.gnn.layers_mp - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(torch.nn.Linear(dim_in, dim_in), torch.nn.BatchNorm1d(dim_in), torch.nn.ReLU(), \
                                    torch.nn.Linear(dim_in, dim_in), torch.nn.BatchNorm1d(dim_in), torch.nn.ReLU()))

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        edge_index = batch.edge_index
        vn_emb = self.virtualnode_embedding(torch.zeros(batch.num_graphs).to(edge_index.dtype).to(edge_index.device))
        for name, module in self.named_children():
            if name in ['virtualnode_embedding', 'mlp_virtualnode_list']: continue
            elif name != 'gnn_layers':
                batch = module(batch)
            else: # add vn support for gnn_layers
                assert name == 'gnn_layers'
                for i, layer in enumerate(module):
                    batch.x = batch.x + vn_emb[batch.batch]
                    batch = layer(batch)
                    if i < cfg.gnn.layers_mp - 1:
                        # add vn_emb + batch
                        vn_emb_temp = global_add_pool(batch.x, batch.batch) + vn_emb
                        if cfg.gnn.vn_residual:
                            vn_emb += F.dropout(self.mlp_virtualnode_list[i](vn_emb_temp), cfg.gnn.dropout, training=self.training)
                        else:
                            vn_emb = F.dropout(self.mlp_virtualnode_list[i](vn_emb_temp), cfg.gnn.dropout, training=self.training)
        return batch
