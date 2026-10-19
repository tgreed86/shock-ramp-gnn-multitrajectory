"""GraphSAGE FeatureNet with the baseline's layer-count and head semantics."""
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def make_activation(name, negative_slope=0.01, elu_alpha=1.):
    factories = {'relu': nn.ReLU, 'leaky_relu': lambda: nn.LeakyReLU(negative_slope),
                 'silu': nn.SiLU, 'gelu': nn.GELU, 'elu': lambda: nn.ELU(elu_alpha), 'identity': nn.Identity}
    if name not in factories:
        raise ValueError(f'Unsupported activation: {name}')
    return factories[name]()


class FeatureNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=128, layers=3, dropout=0.1,
                 use_skip=False, skip_type='block', use_layernorm=False, layernorm_eps=1e-6,
                 activation='relu', activation_negative_slope=0.01, activation_elu_alpha=1.):
        super().__init__()
        if layers < 2:
            raise ValueError('FeatureNet layers must be >=2 (layers-1 message-passing blocks plus the head)')
        self.dropout = dropout
        self.skip_type = skip_type if use_skip else 'none'
        if self.skip_type not in ('none', 'block', 'input', 'both'):
            raise ValueError(f'Invalid skip_type: {skip_type}')
        def act():
            return make_activation(activation, activation_negative_slope, activation_elu_alpha)
        self.block_activation = act()
        dims = [in_channels] + [hidden] * (layers-1)
        self.convs = nn.ModuleList()
        self.block_skip_proj = nn.ModuleList()
        self.block_norms = nn.ModuleList()
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            self.convs.append(SAGEConv(n_in, n_out))
            self.block_skip_proj.append(nn.Linear(n_in, n_out, bias=False)
                if self.skip_type in ('block', 'both') and n_in != n_out else nn.Identity())
            self.block_norms.append(nn.LayerNorm(n_out, eps=layernorm_eps) if use_layernorm else nn.Identity())
        self.input_skip_proj = (nn.Linear(in_channels, hidden, bias=False) if in_channels != hidden else nn.Identity()) \
                                if self.skip_type in ('input', 'both') else None
        self.feat_head = nn.Sequential(nn.Linear(hidden, hidden), act(), nn.Dropout(dropout), nn.Linear(hidden, out_channels))

    def forward(self, x, edge_index):
        h = x
        for conv, skip, norm in zip(self.convs, self.block_skip_proj, self.block_norms):
            h_new = conv(h, edge_index)
            if self.skip_type in ('block', 'both'):
                h_new = h_new + skip(h)
            h = F.dropout(self.block_activation(norm(h_new)), p=self.dropout, training=self.training)
        if self.input_skip_proj is not None:
            h = h + self.input_skip_proj(x)
        return self.feat_head(h)
