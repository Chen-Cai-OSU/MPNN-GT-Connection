# Created by Chen at 12/18/22
from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('vn_cfg')
def vn_cfg(cfg):
    """Deepset VN config
    """
    cfg.dsvn = CN()
    cfg.dsvn.reduction='mean'
    cfg.dsvn.nonlinear='relu'
    cfg.dsvn.n_layers = 1
    cfg.dsvn.batchnorm = True



