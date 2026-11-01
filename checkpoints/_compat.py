"""Internal compatibility loader for the released full-model checkpoints.

These forward methods follow the original MAMBO / MAMBO_cbcic sources.
The saved objects already contain all layers, weights and dataset settings,
so no model constructors or mamba-ssm CUDA extensions are needed here.
"""

import pickle
from types import ModuleType

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def forward(self, x, z):
        x = x * (z * torch.sigmoid(z))
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2Block(nn.Module):
    def forward(self, u):
        z, xbc, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)
        xbc = self.conv1d(xbc.transpose(1, 2)).transpose(1, 2)[:, :u.shape[1]]
        xbc = xbc * torch.sigmoid(xbc)
        x, b, c = xbc.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(*x.shape[:2], self.nheads, self.headdim)
        y = self.ssd(x * dt.unsqueeze(-1), -self.A_log.exp() * dt,
                     b.unsqueeze(2), c.unsqueeze(2))
        y = y + x * self.D.unsqueeze(-1)
        return self.out_proj(self.norm(y.flatten(2), z))

    @staticmethod
    def segsum(x):
        # Preserve the original chunked SSD calculation and its summation order.
        length = x.shape[-1]
        mask = torch.ones(length, length, dtype=torch.bool, device=x.device)
        x = x[..., None].repeat(1, 1, 1, 1, length)
        x = x.masked_fill(~mask.tril(-1), 0).cumsum(dim=-2)
        return x.masked_fill(~mask.tril(), -torch.inf)

    def ssd(self, x, a, b, c):
        size = self.chunk_size
        x = x.reshape(x.shape[0], -1, size, *x.shape[2:])
        b = b.reshape(b.shape[0], -1, size, *b.shape[2:])
        c = c.reshape(c.shape[0], -1, size, *c.shape[2:])
        a = a.reshape(a.shape[0], -1, size, a.shape[2]).permute(0, 3, 1, 2)
        cumulative = a.cumsum(dim=-1)
        diagonal = torch.einsum(
            'bclhn,bcshn,bhcls,bcshp->bclhp', c, b, self.segsum(a).exp(), x
        )
        decay = (cumulative[..., -1:] - cumulative).exp()
        states = torch.einsum('bclhn,bhcl,bclhp->bchpn', b, decay, x)
        states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        across = self.segsum(F.pad(cumulative[..., -1], (1, 0))).exp()[0]
        states = torch.einsum('bhzc,bchpn->bzhpn', across, states)[:, :-1]
        off_diagonal = torch.einsum(
            'bclhn,bchpn,bhcl->bclhp', c, states, cumulative.exp()
        )
        return (diagonal + off_diagonal).flatten(1, 2)


class Mamba2(nn.Module):
    def forward(self, x):
        for block in self.layers:
            x = x + self.resid_drop(block(x))
        return self.final_norm(x)


class DWTProcessor(nn.Module):
    def forward(self, x, drop=True):
        low, bands = self.dwt(x)
        if drop:
            for index in range(self.levels - 4):
                bands[index] = torch.zeros_like(bands[index])
        return low, bands

    def inverse(self, low, bands):
        return self.idwt((low, bands))


class MAMBOStack(nn.Module):
    def forward(self, x):
        low, details = self.waveProcessor(x)
        start = self.waveProcessor.levels - 4
        bands = [getattr(self, f'dpconv{i + 1}')(details[start + i]) for i in range(4)]
        for i in range(3):
            bridge = getattr(self, f'conv{i + 1}')(F.pad(bands[i], (6, 6), mode='reflect'))
            bands[i + 1] = bands[i + 1] + bridge
        bands = [getattr(self, f'mambo_{i + 1}')(band.transpose(1, 2)).transpose(1, 2)
                 for i, band in enumerate(bands)]
        reconstructed = self.waveProcessor.inverse(low, details[:start] + bands)
        return torch.stack([reconstructed, x], dim=1), bands[1]


class CTC(nn.Module):
    def forward(self, x):
        return self.ctpool(self.ctconv(x))


class Conv2dWithConstraint(nn.Conv2d):
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class GCAP(nn.Module):
    def forward(self, x):
        return self.activation(self.bn(self.pointwise(self.depthwise(x))))


class GTCC(nn.Module):
    def forward(self, x):
        return self.avg2(self.tconv(self.avg1(x)))


class Classifier(nn.Module):
    def forward(self, x):
        return self.classifier_1(x)


class Region(nn.Module):
    def forward(self, x):
        left = self.gcap_l(x[:, :, self.listx[0], :])
        right = self.gcap_r(x[:, :, self.listx[1], :])
        center = self.gcap_c(x[:, :, self.listx[2], :])
        return self.gcap_multi(torch.cat([left, center, right], dim=2))


class Region2(nn.Module):
    def forward(self, x):
        center = self.gcap_c(x[:, :, self.listx[0], :])
        outer = self.gcap_o(x[:, :, self.listx[1], :])
        return self.gcap_multi(torch.cat([center, outer], dim=2))


class BetaGate(nn.Module):
    def forward(self, x):
        pooled = torch.cat([self.maxpool(x), self.meanpool(x)], dim=1)
        return self.sigmoid(self.router(pooled))


class CheckpointModel(nn.Module):
    beta_channels = (1, 2, 3, 4, 5)

    def forward(self, x):
        streams, beta = self.MAMBO(x.squeeze(1))
        gate = self.beta_gate(beta[:, self.beta_channels, :])
        features = self.ctconv(streams)
        frequency = self.gcap(features[:, :8])
        raw = features[:, 8:]
        regions = torch.stack([self.region(raw), self.region2(raw)], dim=1)
        raw = torch.einsum('bi,bijkl->bjkl', gate, regions)
        fused = self.gtcc(torch.cat([frequency, raw], dim=1).squeeze(2))
        return self.classifier(fused)


class CBCICModel(CheckpointModel):
    beta_channels = (1, 2, 5, 8, 9)


_NETWORK_CLASSES = dict(
    CTC=CTC, Conv2dWithConstraint=Conv2dWithConstraint, GCAP=GCAP, GTCC=GTCC,
    LinearWithConstraint=LinearWithConstraint, classifier_n=Classifier,
    region=Region, region2=Region2, beta_gate2=BetaGate,
)
_LEGACY_CLASSES = {
    'MAMBO.new_brain2_mambo2_moe2': dict(_NETWORK_CLASSES, base_model_betamoe=CheckpointModel),
    'MAMBO_cbcic2.testmodel101': dict(_NETWORK_CLASSES, base_model_betamoe=CBCICModel),
    'MAMBO.dw_mamba2_att2': {'MAMBO_STACK_n_U': MAMBOStack},
    'MAMBO_cbcic2.dw_mamba2_att2': {'MAMBO_STACK_n_U': MAMBOStack},
    'MAMBO.DWT': {'DWTProcessor': DWTProcessor},
    'MAMBO_cbcic.DWT': {'DWTProcessor': DWTProcessor},
    'MAMBO.mamba2': dict(Mamba2=Mamba2, Mamba2Block=Mamba2Block, RMSNorm=RMSNorm),
    'MAMBO_cbcic.mamba2': dict(Mamba2=Mamba2, Mamba2Block=Mamba2Block, RMSNorm=RMSNorm),
}


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in _LEGACY_CLASSES:
            return _LEGACY_CLASSES[module][name]
        return super().find_class(module, name)


_checkpoint_pickle = ModuleType('checkpoint_pickle')
_checkpoint_pickle.Unpickler = _CheckpointUnpickler


def load_checkpoint(path, device='cpu'):
    """Read this project's trusted full-model .pth files, preserving saved settings."""
    model = torch.load(path, map_location='cpu', pickle_module=_checkpoint_pickle, weights_only=False)
    if not isinstance(model, CheckpointModel):
        raise TypeError(f'{path}: expected a released NeuroMambaNet full-model checkpoint')
    return model.to(device).eval()
