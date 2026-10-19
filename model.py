"""GraphSAGE FeatureNet and an encode-process-decode MeshGraphNet."""
import math

import torch
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


class MeshGraphNetBlock(nn.Module):
    """Update directed edges, sum incoming messages, then update nodes residually."""

    def __init__(self, hidden, make_mlp):
        super().__init__()
        self.edge_mlp = make_mlp(3 * hidden, hidden)
        self.node_mlp = make_mlp(2 * hidden, hidden)

    def forward(self, nodes, edges, edge_index):
        sender, receiver = edge_index
        messages = self.edge_mlp(torch.cat([nodes[sender], nodes[receiver], edges], dim=-1))
        incoming = nodes.new_zeros(nodes.shape)
        incoming.index_add_(0, receiver, messages)
        node_update = self.node_mlp(torch.cat([nodes, incoming], dim=-1))
        # As in MeshGraphNets, aggregate the new messages before adding edge residuals.
        return nodes + node_update, edges + messages


class MeshGraphNet(nn.Module):
    """MeshGraphNets architecture adapted to cell-centered, normalized rate prediction.

    Each directed face edge encodes (sender - receiver) position and its norm,
    divided by a fixed coarse-cell length. Node inputs and rate units are shared
    with FeatureNet. Processor blocks have independent weights; latent MLPs end
    in LayerNorm, while the decoder has a linear output.

    Architecture reference: https://arxiv.org/abs/2010.03409
    """

    def __init__(self, in_channels, out_channels, hidden=128, message_passing_steps=15,
                 mlp_layers=2, dropout=0.1, layernorm_eps=1e-6, activation='relu',
                 activation_negative_slope=0.01, activation_elu_alpha=1., length_scale=1.):
        super().__init__()
        for name, value in [('hidden', hidden), ('message_passing_steps', message_passing_steps),
                            ('mlp_layers', mlp_layers)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'MeshGraphNet {name} must be a positive integer')
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError('MeshGraphNet length_scale must be finite and positive')
        self.length_scale = float(length_scale)

        def mlp(n_in, n_out, *, normalize=True):
            modules = []
            for _ in range(mlp_layers):
                modules.extend([nn.Linear(n_in, hidden),
                                make_activation(activation, activation_negative_slope, activation_elu_alpha),
                                nn.Dropout(dropout)])
                n_in = hidden
            modules.append(nn.Linear(hidden, n_out))
            if normalize:
                modules.append(nn.LayerNorm(n_out, eps=layernorm_eps))
            return nn.Sequential(*modules)

        self.node_encoder = mlp(in_channels, hidden)
        self.edge_encoder = mlp(3, hidden)
        self.processor = nn.ModuleList([MeshGraphNetBlock(hidden, mlp) for _ in range(message_passing_steps)])
        self.decoder = mlp(hidden, out_channels, normalize=False)

    def edge_features(self, centers, edge_index):
        relative = (centers[edge_index[0]] - centers[edge_index[1]]) / self.length_scale
        return torch.cat([relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)], dim=-1)

    def forward(self, x, edge_index, centers):
        nodes = self.node_encoder(x)
        edges = self.edge_encoder(self.edge_features(centers, edge_index).to(dtype=x.dtype))
        for block in self.processor:
            nodes, edges = block(nodes, edges, edge_index)
        return self.decoder(nodes)
