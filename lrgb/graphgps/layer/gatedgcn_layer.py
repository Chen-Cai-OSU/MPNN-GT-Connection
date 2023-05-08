import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_layer


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """
        GatedGCN layer
        Residual Gated Graph ConvNets
        https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim), nn.ReLU(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.bn_node_x = nn.BatchNorm1d(out_dim)
        self.bn_edge_e = nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        self.residual = residual
        self.e = None

    def forward(self, *args):
        if len(args)==1:
            batch = args[0]
            x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
            overwrite_edge = True
        elif len(args)==4:
            x, e, edge_index, batch = args
            overwrite_edge = False
        else:
            raise NotImplementedError


        """
        x               : [n_nodes, in_dim]
        e               : [n_edges, in_dim]
        edge_index      : [2, n_edges]
        """
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE else None

        x, e = self.propagate(edge_index,
                              Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce,
                              e=e, Ax=Ax,
                              PE=pe_LapPE)

        x = self.bn_node_x(x)
        e = self.bn_edge_e(e)

        x = F.relu(x)
        e = F.relu(e)

        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        if overwrite_edge:
            batch.x = x
            batch.edge_attr = e
            return batch
        else:
            return x, e

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        """
        {}x_i           : [n_edges, out_dim]
        {}x_j           : [n_edges, out_dim]
        {}e             : [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        # Handling for Equivariant and Stable PE using LapPE
        # ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
        if self.EquivStablePE:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)  # the MLP is 1 dim --> hidden_dim --> 1 dim
            sigma_ij = sigma_ij * r_ij

        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        """
        sigma_ij        : [n_edges, out_dim]  ; is the output from message() function
        index           : [n_edges]
        {}x_j           : [n_edges, out_dim]
        """
        dim_size = Bx.shape[0]  # or None ??   <--- Double check this

        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size,
                                   reduce='sum')

        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size,
                                     reduce='sum')

        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        """
        aggr_out        : [n_nodes, out_dim] ; is the output from aggregate() function after the aggregation
        {}x             : [n_nodes, out_dim]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class HeteroGatedGCNLayer(pyg_nn.conv.MessagePassing):
    def __init__(self, in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.model1 = GatedGCNLayer(in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs)
        self.model2 = GatedGCNLayer(in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs)
        self.model3 = GatedGCNLayer(in_dim, out_dim, dropout, residual,
                 equivstable_pe=False, **kwargs)

    def forward(self, batch):
        indices0 = batch.edge_type == 0
        x0, e0 = self.model1(batch.x, batch.edge_attr[indices0, :], batch.edge_index[:, indices0], batch)
        # graph node - vn
        indices1 = batch.edge_type == 1
        x1, e1 = self.model2(batch.x, batch.edge_attr[indices1, :], batch.edge_index[:, indices1], batch)

        # vn - graph node
        indices2 = batch.edge_type == 2
        x2, e2 = self.model3(batch.x, batch.edge_attr[indices2, :], batch.edge_index[:, indices2], batch)

        batch.x = x0 + x1 + x2
        batch.edge_attr[indices0, :] = e0
        batch.edge_attr[indices1, :] = e1
        batch.edge_attr[indices2, :] = e2
        return batch


class GatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GatedGCNLayer(in_dim=layer_config.dim_in,
                                   out_dim=layer_config.dim_out,
                                   dropout=0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
                                   residual=False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
                                   **kwargs)

    def forward(self, batch):
        return self.model(batch)


register_layer('gatedgcnconv', GatedGCNGraphGymLayer)

class HeteroGatedGCNGraphGymLayer(nn.Module):
    """GatedGCN layer.
    Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        kw = {'in_dim': layer_config.dim_in,
              'out_dim': layer_config.dim_out,
              'dropout': 0.,  # Dropout is handled by GraphGym's `GeneralLayer` wrapper
              'residual': False,  # Residual connections are handled by GraphGym's `GNNStackStage` wrapper
              }
        kw = {**kw, **kwargs}

        self.model1 = GatedGCNLayer(**kw)
        self.model2 = GatedGCNLayer(**kw)
        self.model3 = GatedGCNLayer(**kw)

    def forward(self, batch):
        # graph nodes - graph nodes
        indices = batch.edge_type == 0
        import pdb; pdb.set_trace()
        batch_tmp1 = self.model1(batch.x, batch.edge_attr[indices, :], batch.edge_index[:, indices])

        # graph node - vn
        indices = batch.edge_type == 1
        batch_tmp2 = self.model2(batch.x, batch.edge_attr[indices, :], batch.edge_index[:, indices])

        # vn - graph node
        indices = batch.edge_type == 2
        batch_tmp3 = self.model3(batch.x, batch.edge_attr[indices, :], batch.edge_index[:, indices])

        batch.x = batch_tmp1.x + batch_tmp2.x + batch_tmp3.x
        batch.edge_attr = batch_tmp1.edge_attr + batch_tmp2.edge_attr + batch_tmp3.edge_attr
        return batch

register_layer('hetero_gatedgcnconv', HeteroGatedGCNGraphGymLayer)
