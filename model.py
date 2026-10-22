"""GraphSAGE, MeshGraphNet, and conservative FluxGraphNet predictors."""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


RAMP_QUANTITIES = ('density', 'x_momentum', 'y_momentum', 'energy')


def resolve_ramp_boundary_channels(model_config):
    """Convert named conserved quantities to channels allowed wall exchange.

    Keep the original channel-mask configuration for existing checkpoints.
    Named quantities specify zero boundary flux, not zero interior transport.
    """
    key = 'ramp_conserved_quantities'
    if key not in model_config:
        return model_config.get('ramp_boundary_source_channels', (1, 2))
    if 'ramp_boundary_source_channels' in model_config:
        raise ValueError('Specify model.ramp_conserved_quantities or model.ramp_boundary_source_channels, not both')
    quantities = model_config[key]
    if (not isinstance(quantities, list)
            or any(not isinstance(q, str) or q not in RAMP_QUANTITIES for q in quantities)):
        raise ValueError(f'model.{key} must be a list drawn from {list(RAMP_QUANTITIES)}; [] conserves none')
    if len(set(quantities)) != len(quantities):
        raise ValueError(f'model.{key} must not contain duplicate quantities')
    return tuple(i for i, name in enumerate(RAMP_QUANTITIES) if name not in quantities)


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
        self._mlp_options = dict(hidden=hidden, mlp_layers=mlp_layers, dropout=dropout,
            layernorm_eps=layernorm_eps, activation=activation,
            activation_negative_slope=activation_negative_slope, activation_elu_alpha=activation_elu_alpha)

        mlp = self.make_mlp
        self.node_encoder = mlp(in_channels, hidden)
        self.edge_encoder = mlp(3, hidden)
        self.processor = nn.ModuleList([MeshGraphNetBlock(hidden, mlp) for _ in range(message_passing_steps)])
        self.decoder = mlp(hidden, out_channels, normalize=False)

    def make_mlp(self, n_in, n_out, *, normalize=True):
        opts = self._mlp_options
        hidden = opts['hidden']
        modules = []
        for _ in range(opts['mlp_layers']):
            modules.extend([nn.Linear(n_in, hidden),
                            make_activation(opts['activation'], opts['activation_negative_slope'],
                                            opts['activation_elu_alpha']),
                            nn.Dropout(opts['dropout'])])
            n_in = hidden
        modules.append(nn.Linear(hidden, n_out))
        if normalize:
            modules.append(nn.LayerNorm(n_out, eps=opts['layernorm_eps']))
        return nn.Sequential(*modules)

    def edge_features(self, centers, edge_index):
        relative = (centers[edge_index[0]] - centers[edge_index[1]]) / self.length_scale
        return torch.cat([relative, torch.linalg.vector_norm(relative, dim=-1, keepdim=True)], dim=-1)

    def encode_process(self, x, edge_index, centers):
        nodes = self.node_encoder(x)
        edges = self.edge_encoder(self.edge_features(centers, edge_index).to(dtype=x.dtype))
        for block in self.processor:
            nodes, edges = block(nodes, edges, edge_index)
        return nodes, edges

    def forward(self, x, edge_index, centers):
        nodes, _ = self.encode_process(x, edge_index, centers)
        return self.decoder(nodes)


class FluxGraphNet(MeshGraphNet):
    """MeshGraphNet with a face-flux decoder and separate exterior heads.

    Flux densities are in units sigma * length_scale / dt_ref; the returned
    node rate is in sigma / dt_ref units, as for MeshGraphNet. Every interior
    face contributes one transport with opposite signs to its two cells,
    divided by their respective areas. Boundary fluxes are outward-positive.
    """

    def __init__(self, in_channels, out_channels, *, use_open_boundary_source=True,
                 use_ramp_boundary_source=True, open_boundary_source_channels=(0, 1, 2, 3),
                 ramp_boundary_source_channels=(1, 2), **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        hidden = self._mlp_options['hidden']
        self.decoder = self.make_mlp(3 * hidden, out_channels, normalize=False)
        self.open_boundary_head = (self.make_mlp(hidden + 2, out_channels, normalize=False)
                                   if use_open_boundary_source else None)
        self.ramp_boundary_head = (self.make_mlp(hidden + 2, out_channels, normalize=False)
                                   if use_ramp_boundary_source else None)
        for name, channels in [('open', open_boundary_source_channels), ('ramp', ramp_boundary_source_channels)]:
            mask = torch.zeros(out_channels)
            if channels is not None:
                for channel in channels:
                    if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < out_channels:
                        raise ValueError(f'{name}_boundary_source_channels must index output channels')
                    mask[channel] = 1
            self.register_buffer(f'{name}_boundary_mask', mask, persistent=False)

    def predict_rate_and_flux(self, x, edge_index, centers, geometry):
        """Return rate and integrated interior/exterior transports.

        Transports are in sigma * length_scale**2 / dt_ref units and align
        with geometry.forward_edges and geometry.boundary_owner. Multiply by
        sigma * length_scale**2 / dt_ref to recover physical transport rates.
        These are the raw predictions, before the engine's shared limiter.
        """
        nodes, edges = self.encode_process(x, edge_index, centers)
        fwd, rev = geometry.forward_edges, geometry.reverse_edges
        src, dst = edge_index[:, fwd]
        forward = torch.cat([nodes[src], nodes[dst], edges[fwd]], dim=-1)
        reverse = torch.cat([nodes[dst], nodes[src], edges[rev]], dim=-1)
        decoded = self.decoder(torch.cat([forward, reverse]))
        flux = .5 * (decoded[:len(fwd)] - decoded[len(fwd):])
        transport = flux * (geometry.face_length.to(x) / self.length_scale)[:, None]
        balance = x.new_zeros((len(x), transport.shape[1]))
        balance.index_add_(0, src, -transport)
        balance.index_add_(0, dst, transport)

        owner = geometry.boundary_owner
        boundary_input = torch.cat([nodes[owner], geometry.boundary_normal.to(x)], dim=-1)
        boundary_flux = x.new_zeros((len(owner), transport.shape[1]))
        for head, selected, mask in (
                (self.open_boundary_head, geometry.boundary_open, self.open_boundary_mask),
                (self.ramp_boundary_head, ~geometry.boundary_open, self.ramp_boundary_mask)):
            if head is not None:
                boundary_flux[selected] = head(boundary_input[selected]) * mask
        boundary_transport = boundary_flux * (geometry.boundary_length.to(x) / self.length_scale)[:, None]
        balance.index_add_(0, owner, -boundary_transport)
        rate = balance / (geometry.area.to(x) / self.length_scale**2)[:, None]
        return rate, transport, boundary_transport

    def forward(self, x, edge_index, centers, geometry):
        return self.predict_rate_and_flux(x, edge_index, centers, geometry)[0]
