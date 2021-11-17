"""model.py"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import json
import numpy as np


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def compute_kl(z_1, z_2, logvar_1, logvar_2):
    var_1 = logvar_1.exp()
    var_2 = logvar_2.exp()
    return var_1/var_2 + ((z_2-z_1)**2)/var_2 - 1 + logvar_2 - logvar_1


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3, pcl=False, gvae=False, mlvae=False):
        super(BetaVAE_H, self).__init__()
        self.gvae = gvae
        self.mlvae = mlvae
        self.pcl = pcl
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim if pcl else z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, return_z=False):
        distributions = self._encode(x)
        if self.pcl:
            return None, distributions, None
        else:
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            if self.gvae or self.mlvae:
                mu1, mu2, logvar1, logvar2 = mu[::2].clone(), mu[1::2].clone(), logvar[::2].clone(), logvar[1::2].clone()
                kl_per_point = compute_kl(mu1, mu2, logvar1, logvar2)
                if self.gvae:
                    new_mu = 0.5 * mu1 + 0.5 * mu2
                    new_logvar = (0.5 * logvar1.exp() + 0.5 * logvar2.exp()).log()
                else:
                    var_1 = logvar1.exp()
                    var_2 = logvar2.exp()
                    new_var = 2*var_1 * var_2 / (var_1 + var_2)
                    new_mu = (mu1/var_1 + mu2/var_2)*new_var*0.5
                    new_logvar = new_var.log()
                histogram = []
                for x in kl_per_point:
                    x_normal = (x - x.min()) / (x.max() - x.min())
                    indices = torch.floor(2 * x_normal)
                    histogram.append(torch.clamp(indices, 0, 1).int())
                histogram = torch.stack(histogram)
                mask = torch.eq(histogram, torch.ones_like(histogram))
                mu[::2] = torch.where(mask, mu1, new_mu)
                logvar[::2] = torch.where(mask, logvar1, new_logvar)
                mu[1::2] = torch.where(mask, mu2, new_mu)
                logvar[1::2] = torch.where(mask, logvar2, new_logvar)
            z = reparametrize(mu, logvar)
            x_recon = self._decode(z)

            if return_z:
                return x_recon, mu, logvar, z
            else:
                return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()