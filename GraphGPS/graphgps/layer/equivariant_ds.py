# Created by Chen at 12/17/22
# equivariant deepsets layer
# Permutation Equivariant and Permutation Invariant layers, as described in the
# paper Deep Sets, by Zaheer et al. (https://arxiv.org/abs/1703.06114)
# https://github.com/manzilzaheer/DeepSets/blob/master/SetExpansion/model.py
from collections import OrderedDict

import math
import torch
from torch import nn
from torch.nn import init


class InvLinear(nn.Module):
    r"""Permutation invariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """

    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super(InvLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert reduction in ['mean', 'sum', 'max', 'min'], \
            '\'reduction\' should be \'mean\'/\'sum\'\'max\'/\'min\', got {}'.format(reduction)
        self.reduction = reduction

        self.beta = nn.Parameter(torch.Tensor(self.in_features,
                                              self.out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.beta)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.beta)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to a vector y of dimension out_features,
        through a permutation invariant linear transformation of the form:
            $y = \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N vectors of dimension out_features (tensor with shape (N, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        y = torch.zeros(N, self.out_features).to(device)
        if mask is None:
            mask = torch.ones(N, M).byte().to(device)

        if self.reduction == 'mean':
            sizes = mask.float().sum(dim=1).unsqueeze(1)
            Z = X * mask.unsqueeze(2).float()
            y = (Z.sum(dim=1) @ self.beta) / sizes

        elif self.reduction == 'sum':
            Z = X * mask.unsqueeze(2).float()
            y = Z.sum(dim=1) @ self.beta

        elif self.reduction == 'max':
            Z = X.clone()
            Z[~mask] = float('-Inf')
            y = Z.max(dim=1)[0] @ self.beta

        else:  # min
            Z = X.clone()
            Z[~mask] = float('Inf')
            y = Z.min(dim=1)[0] @ self.beta

        if self.bias is not None:
            y += self.bias

        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, reduction={}'.format(
            self.in_features, self.out_features,
            self.bias is not None, self.reduction)


class EquivLinear(InvLinear):
    r"""Permutation equivariant linear layer.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
        reduction: Permutation invariant operation that maps the input set into a single
            vector. Currently, the following are supported: mean, sum, max and min.
    """

    def __init__(self, in_features, out_features, bias=True, reduction='mean'):
        super(EquivLinear, self).__init__(in_features, out_features,
                                          bias=bias, reduction=reduction)

        self.alpha = nn.Parameter(torch.Tensor(self.in_features,
                                               self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        super(EquivLinear, self).reset_parameters()
        if hasattr(self, 'alpha'):
            init.xavier_uniform_(self.alpha)

    def forward(self, X, mask=None):
        r"""
        Maps the input set X = {x_1, ..., x_M} to the output set
        Y = {y_1, ..., y_M} through a permutation equivariant linear transformation
        of the form:
            $y_i = \alpha x_i + \beta reduction(X) + bias$
        Inputs:
        X: N sets of size at most M where each element has dimension in_features
           (tensor with shape (N, M, in_features))
        mask: binary mask to indicate which elements in X are valid (byte tensor
            with shape (N, M) or None); if None, all sets have the maximum size M.
            Default: ``None``.
        Outputs:
        Y: N sets of same cardinality as in X where each element has dimension
           out_features (tensor with shape (N, M, out_features))
        """
        N, M, _ = X.shape
        device = X.device
        if mask is None:
            mask = torch.ones(N, M, dtype=torch.bool).to(device)

        Y = torch.zeros(N, M, self.out_features).to(device)
        h_inv = super(EquivLinear, self).forward(X, mask=mask)
        Y[mask] = (X @ self.alpha + h_inv.unsqueeze(1))[mask]

        return Y


class EquivariantDS(nn.Module):
    def __init__(self, d, n_layers, reduction='mean', nonlinear='relu', bn=True):
        super(EquivariantDS, self).__init__()
        self.d = d
        self.n_layers = n_layers
        self.bn = bn
        layers = []
        for i in range(n_layers):
            layers.append((f'EquivLinear-{i}', EquivLinear(d, d, reduction=reduction)))
            layers.append((f'{nonlinear}-{i}', self.set_nonlinear(nonlinear)))
        self.ds = nn.Sequential(OrderedDict(layers))
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(d) for _ in range(n_layers)])

    def set_nonlinear(self, nonlinear):
        if nonlinear == 'relu':
            nl = nn.ReLU()
        elif nonlinear == 'elu':
            nl = nn.ELU()
        else:
            nl = nn.Identity()
        return nl

    def forward(self, batch, mask=None):
        for i in range(self.n_layers):
            batch = self.ds[2 * i](batch, mask=mask)
            batch = self.ds[2 * i + 1](batch)
            if self.bn:
                batch = torch.permute(batch, (0, 2, 1))
                batch = self.bn_layers[i](batch)
                batch = torch.permute(batch, (0, 2, 1))
        return batch


if __name__ == "__main__":
    x = torch.rand((2, 2, 4))
    permute_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    x_prime = permute_matrix @ x
    layer = EquivariantDS(4, 3)  # EquivLinear(4, 10,)
    y = layer(x)
    y_prime = layer(x_prime)
    print(torch.max(torch.abs(y_prime - permute_matrix @ y)))
