import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from scripts.model import BetaVAE_H as BetaVAE
import numpy as np
import imageio
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss

def compute_cross_ent_normal(mu, logvar):
    return 0.5 * (mu**2 + torch.exp(logvar)) + np.log(np.sqrt(2 * np.pi))

def compute_ent_normal(logvar):
    return 0.5 * (logvar + np.log(2 * np.pi * np.e))

def compute_sparsity(mu, normed=True):
    # assume couples, compute normalized sparsity
    diff = mu[::2] - mu[1::2]
    if normed:
        norm = torch.norm(diff, dim=1, keepdim=True)
        norm[norm == 0] = 1  # keep those that are same, dont divide by 0
        diff = diff / norm
    return torch.mean(torch.abs(diff))


class Solver(object):
    def __init__(self, args, data_loader=None):
        self.ckpt_dir = args.ckpt_dir
        self.output_dir = args.output_dir
        self.data_loader = data_loader
        self.dataset = args.dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.nc = args.num_channel
        self.decoder_dist = 'bernoulli'
        self.beta = args.beta  # for kl to normal
        self.gamma = args.gamma  # for kl to laplace
        self.rate_prior = args.rate_prior * torch.ones(
            1, requires_grad=False, device=self.device)
        params = []

        # for adam
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.normal_dist = torch.distributions.normal.Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device))

        self.net = BetaVAE(self.z_dim, self.nc).to(self.device)
        self.optim = optim.Adam(
            params + list(self.net.parameters()), lr=self.lr,
            betas=(self.beta1, self.beta2))

        self.ckpt_name = args.ckpt_name
        if False and self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.log_step = args.log_step
        self.save_step = args.save_step

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = - torch.log(rate_prior / 2) + rate_prior * sigma *\
             np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
             rate_prior * mean * (
                     1 - 2 * self.normal_dist.cdf(mean / sigma))
        return ce

    def compute_cross_ent_combined(self, mu, logvar):
        normal_entropy = compute_ent_normal(logvar)
        cross_ent_normal = compute_cross_ent_normal(mu, logvar)
        # assuming couples, do Laplace both ways
        mu0 = mu[::2]
        mu1 = mu[1::2]
        logvar0 = logvar[::2]
        logvar1 = logvar[1::2]
        rate_prior0 = self.rate_prior
        rate_prior1 = self.rate_prior
        cross_ent_laplace = (
            self.compute_cross_ent_laplace(mu0 - mu1, logvar0, rate_prior0) +
            self.compute_cross_ent_laplace(mu1 - mu0, logvar1, rate_prior1))
        return [x.sum(1).mean(0, True) for x in [normal_entropy,
                                                 cross_ent_normal,
                                                 cross_ent_laplace]]

    def train(self, writer):
        self.net_mode(train=True)
        out = False  # whether to exit training loop
        failure = False  # whether training was stopped
        running_loss = np.zeros(5)
        running_mean_vars = np.zeros(self.z_dim)  # variance of means
        running_var_means = np.zeros(self.z_dim)  # means of variances
        running_other = np.zeros(4)
        log = open(os.path.join(self.output_dir, 'log.csv'), 'a', 1)
        log.write('negLLH,H(Q),H(Q;P_N),H(Q;P_L),Total\n')
        log_mean_vars = open(
            os.path.join(self.output_dir, 'mean_vars.csv'), 'a', 1)
        log_mean_vars.write('Total,' + ','.join([str(v) for v in np.arange(
            self.z_dim)]) + '\n')
        log_var_means = open(
            os.path.join(self.output_dir, 'var_means.csv'), 'a', 1)
        log_var_means.write('Total,' + ','.join([str(v) for v in np.arange(
            self.z_dim)]) + '\n')
        log_other = open(os.path.join(self.output_dir, 'other.csv'), 'a', 1)
        log_other.write('|L1|,rate_prior,H(Q_0),H(Q_1)\n')

        while not out:
            for x, _ in self.data_loader:  # don't use label
                x = Variable(x.to(self.device))
                x_recon, mu, logvar = self.net(x)
                # mu shape: Batch x latent_dim
                mean_vars = torch.var(mu, dim=0)
                var_means = torch.mean(torch.exp(logvar), dim=0)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)

                if torch.isnan(recon_loss):
                    print('cancel because of nan in loss, iter',
                          self.global_iter)
                    failure = True
                    out = True
                    break

                # train both ways
                [normal_entropy, cross_ent_normal, cross_ent_laplace
                 ] = self.compute_cross_ent_combined(mu, logvar)
                vae_loss = 2 * recon_loss
                kl_normal = cross_ent_normal - normal_entropy
                kl_laplace = cross_ent_laplace - normal_entropy
                vae_loss = vae_loss + self.beta * kl_normal
                vae_loss = vae_loss + self.gamma * kl_laplace

                # logging
                running_loss[0] += recon_loss.item()
                running_loss[4] += vae_loss.item()
                sparsity = compute_sparsity(mu, normed=False)
                running_other[0] += sparsity.item()
                running_other[1] += self.rate_prior.item()
                running_mean_vars += mean_vars.detach().cpu().numpy()
                running_var_means += var_means.detach().cpu().numpy()

                # train both ways
                running_loss[1] = normal_entropy.item()
                running_loss[2] += cross_ent_normal.item()
                running_loss[3] += cross_ent_laplace.item()

                self.optim.zero_grad()
                vae_loss.backward()
                self.optim.step()

                self.global_iter += 1
                if self.global_iter % self.log_step == 0:
                    running_loss /= self.log_step
                    log.write(','.join('%.6f' % r for r in running_loss) + '\n')
                    if writer is not None:
                        for loss, name in zip(running_loss, [
                            'rconst', 'normal entropy', 'coss_ent_normal',
                            'cross_ent_lap', 'vae_loss']):
                            writer.add_scalar('/loss/' + name, loss, self.global_iter)

                        writer.add_image('inputs', make_grid(x), self.global_iter)
                        writer.add_image(
                            'reconsts', make_grid(x_recon.clamp_(0, 1)), self.global_iter)
                    running_mean_vars /= self.log_step
                    log_mean_vars.write(
                        '%.6f,' % np.sum(running_mean_vars) + ','.join(
                            ['%.6f' % v for v in running_mean_vars]) + '\n')
                    running_var_means /= self.log_step
                    log_var_means.write(
                        '%.6f,' % np.sum(running_var_means) + ','.join(
                            ['%.6f' % v for v in running_var_means]) + '\n')
                    running_other /= self.log_step
                    log_other.write(
                        ','.join('%.6f' % r for r in running_other) + '\n')

                    # early stopping
                    if self.dataset == 'dsprites' and self.global_iter > \
                            100000 and running_loss[0] > 100:
                        print('cancel because LLH > 100 after 100k steps')
                        failure = True
                        out = True
                        break

                    running_loss = np.zeros_like(running_loss)
                    running_mean_vars = np.zeros(self.z_dim)
                    running_var_means = np.zeros(self.z_dim)
                    running_other = np.zeros_like(running_other)

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        # in the end traverse anyway
        try:
            self.traverse()
        except RuntimeError:
            print('skip the traversal because of CUDA OOM.')

        if failure:
            shutil.rmtree(self.ckpt_dir)

        return failure

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(
                file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def traverse(self, lowest_prob=0.05, n_steps=40):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        # get some random samples
        random_img =  torch.tensor(self.data_loader.dataset.__getitem__(rand_idx)[0][None])
        if len(random_img.shape) == 3:
            random_img = random_img[None]
        random_img = Variable(random_img.to(self.device), volatile=True)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(torch.rand(1, self.z_dim).to(self.device), volatile=True)

        # some data based samples
        if self.dataset == 'dsprites':
            ids = [87040, 332800, 578560]
            names = ['fixed_square', 'fixed_ellipse', 'fixed_heart']
        elif self.dataset == 'yvos':
            ids = [0, 10022, 20032]
            names = ['fixed_img1' ,'fixed_img2', 'fixed_img3']
        else:
            ids = [0, 1002, 2002]
            names = ['fixed_img1' ,'fixed_img2', 'fixed_img3']

        Z = {'random_img':random_img_z, 'random_z':random_z}
        # infer latents z from image
        for id_i, name in zip(ids, names):
            fixed_img = torch.tensor(self.data_loader.dataset.__getitem__(id_i)[0][None])
            if len(fixed_img.shape) == 3:
                fixed_img = fixed_img[None]
            fixed_img = Variable(fixed_img.to(self.device), volatile=True)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
            Z[name] = fixed_img_z

        # do latent walk along axis
        interpolation = torch.linspace(lowest_prob, 1 - lowest_prob, steps=n_steps)
        interpolation = interpolation.to(self.device)
        # fit gaussian to estimate step sizes in latent walk
        n_total = 0
        out, labels = [], []
        for b, l in self.data_loader:
            labels.append(l)
            out.append(encoder(b.to(self.device)))
            n_total += b.shape[0]
            if n_total > 10000:
                break
        labels = torch.cat(labels, dim=0)#[::2]  # only starting points
        zs = torch.cat(out)
        normals = [Normal(zs[:, i].mean(), zs[:, i].std()) for i in range(self.z_dim)]

        # generate gif
        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row, normal_distribution in zip(range(self.z_dim), normals):
                z = z_ori.clone()
                for val in normal_distribution.icdf(interpolation):
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)

        # write to file
        output_dir = os.path.join(self.output_dir, 'traversals')
        os.makedirs(output_dir, exist_ok=True)
        gifs = torch.cat(gifs).cpu()
        gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
        for i, key in enumerate(Z.keys()):
            grids = []
            for j, val in enumerate(interpolation):
                grids.append((make_grid(gifs[i, j], pad_value=1, nrow=10).permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            imageio.mimsave(os.path.join(output_dir, key + '.gif'), grids)

        # plot latent embeddings
        if self.dataset != 'natural':
            plot_latents(zs, labels, output_dir, z_dim=self.z_dim)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()


def plot_latents(zs, labels, out_dir, show=False, z_dim=10):
    mus = zs.detach().cpu().numpy()[:, :z_dim]
    logvars = zs.detach().cpu().numpy()[:, z_dim:]
    labels = labels.detach().cpu().numpy()
    num_factors = labels.shape[1]

    sorting = np.argsort(np.var(mus, 0))[::-1]

    plt.figure(figsize=(4 * num_factors, 16))
    for i in range(z_dim // 2):
        ind_x = sorting[0+i*2]
        ind_y = sorting[1+i*2]
        for j in range(num_factors):
            plt.subplot(5, num_factors, 1 + j + i * num_factors)
            plt.scatter(
                mus[:, ind_x],
                mus[:, ind_y],
                s=.1, c=labels[:, j]
            )
            plt.xlabel('Latent %s, mean_var=%.4f' % (
                ind_x, np.mean(np.exp(logvars[:, ind_x]))))
            plt.ylabel('Latent %s, mean_var=%.4f' % (
                ind_y, np.mean(np.exp(logvars[:, ind_y]))))
            plt.colorbar(label='Ground Truth Factor %s' % j)
            plt.grid()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(out_dir, 'latent_embedding.png'))
        plt.clf()
