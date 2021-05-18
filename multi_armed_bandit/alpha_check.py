# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:29:45 2021

@author: 44759
"""

import os
#os.chdir('D:\\Projects\\renyi_bound_analytics\\renyibounds\\multi_armed_bandit')
import util_chesl as ut
import numpy as np
from tqdm import tqdm
import autograd.numpy as agnp
import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
from autograd.misc.optimizers import adam
import wandb
import autograd.numpy.random as agnpr
import torch.nn as nn
import torch.optim as optim
import torch
from torch.distributions.normal import Normal
import math

def_config = {'round': 1000,
              'bou': 'reny_bms',
              'learning_rate': 5e-1,
              'bound': 'Renyi',
              'alpha': 2,
              'num_iters': 100,
              'simulations': 1,
              'prior_mu': 15,
              'prior_sigma': 1.0,
              'likelihood_sigma': 1.0,
              'gen_proc_mode1': 0,
              'gen_proc_mode2': 20,
              'gen_proc_std1': 5,
              'gen_proc_std2': 5,
              'gen_proc_weight1': 0.7,
              'num_obs': 5000,
              'mc_samples': 1000,
              'run': 1,
              }

wandb.init(project='renyi_0.5', config=def_config)
config = wandb.config
device = torch.device("cpu")
np.random.seed(1)


def generate_obs():
    obs = []
    for j in range(config['num_obs']):
        y1 = np.random.normal(config['gen_proc_mode1'], config['gen_proc_std1'])
        y2 = np.random.normal(config['gen_proc_mode2'], config['gen_proc_std2'])
        w = np.random.binomial(1, config['gen_proc_weight1'], 1)
        reward = w*y1 + (1-w)*y2
        obs.append(reward.item())
    return obs


obs = generate_obs()

policy = ut.VariationalPolicy(config['prior_mu'], config['prior_sigma'])
optimize_policy = optim.Adam(policy.parameters(), lr=config['learning_rate'])

def compute_policy_loss(params):
    def unpack_params(params):
        mu, log_sigma = params[0], params[1]
        return mu, torch.exp(log_sigma)

    def log_q(samples, mu, sigma):
        log_q = -0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (samples - mu) **2 / sigma
        return log_q

    # Q samples
    mu, sigma = unpack_params(params)
    #print("unpacked params", mu, sigma)
    samples = (torch.randn(config['mc_samples']) * torch.sqrt(sigma) + mu)
    # print("samples", samples)

    p0 = Normal(torch.tensor([config['prior_mu']],  dtype=torch.float32), torch.tensor([config['prior_sigma']],  dtype=torch.float32))
    factor = Normal(torch.tensor([obs],  dtype=torch.float32), torch.tensor([config['likelihood_sigma']],  dtype=torch.float32))
    noise = Normal(torch.tensor([0.0],  dtype=torch.float32), torch.tensor([1.0],  dtype=torch.float32))


    logp0 = p0.log_prob(samples)
    logfactor = torch.sum(factor.log_prob(samples.reshape(-1, 1)), axis=1)


    v_noise = torch.exp(noise.log_prob(1.0))
    # =============================================================================
    #         logp0 = log_prior(samples,1.0)
    # =============================================================================
    logq = log_q(samples, mu, v_noise)

    if config['bound'] == 'Elbo':
        # =============================================================================
        #             lower_bound =  -(agnp.mean(gaussian_entropy(sigma) + (logp0+ logfactor)))
        # =============================================================================
        KL = torch.sum(-0.5 * torch.log(2 * math.pi * v_noise) - 0.5 * (mu ** 2 + sigma) / v_noise) - \
             torch.sum(-0.5 * torch.log(2 * math.pi * sigma * np.exp(1)))
        lower_bound = -(torch.mean(logfactor) + KL)  # kl(mu, sigma))

    elif config['bound'] == 'Renyi':
        # print('ps',logp0 )
        # print('po',logfactor  )
        # print('q',logq  )
        logF = logp0 + logfactor - logq
        # print('logF',logF)
        logF = (1 - config['alpha']) * logF
        # print('logF scaled',logF)
        lower_bound = -(torch.logsumexp(logF, 0) - torch.logsumexp(logq, 0))
        # print('Lower bound',lower_bound)
        lower_bound = lower_bound / (config['mc_samples'] * (1 - config['alpha']))
        # print('Lower bound scaled',lower_bound)

        # print('Renyi-negmax', agnp.min(logF))
        # print('Renyi', lower_bound)
        # print('Renyi-half', -2*agnp.sum(0.5*logq+0.5*(logp0+logfactor)))
        KL = torch.sum(-0.5 * torch.log(2 * math.pi * v_noise) - 0.5 * (mu ** 2 + sigma) / v_noise) - \
             torch.sum(-0.5 * torch.log(2 * math.pi * sigma * np.exp(1)))
        elbo = -(torch.mean(logfactor) + KL)  # kl(mu, sigma))
        # print('Elbo',elbo)
        # print('Renyi-max', agnp.max(logq - (logp0 + logfactor)))

    elif config['bound'] == 'Renyi-half':
        lower_bound = -2 * torch.sum(0.5 * logq + 0.5 * (logp0 + logfactor))

    elif config['bound'] == 'Max':
        logF = logq - (logp0 + logfactor)
        lower_bound = -torch.max(logF)

    elif config['bound'] == 'negMax':
        logF = (logp0 + logfactor) - logq
        lower_bound = -torch.max(logF)

    return lower_bound, mu.detach().item(), sigma.detach().item()


def learn():
    # Update policy
    for _ in range(config['num_iters']):
        params = nn.utils.parameters_to_vector(list(policy.parameters())).to(device, non_blocking=True)

        optimize_policy.zero_grad()
        loss_policy, mu, sigma = compute_policy_loss(params)
        print("loss", loss_policy.detach().item(), "mu", mu, "sigma", sigma)
        loss_policy.backward()
        optimize_policy.step()


learn()