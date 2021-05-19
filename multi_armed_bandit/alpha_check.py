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
import torch.distributions as D
import math

def_config = {'learning_rate': 5e-1,
              'bound': 'Renyi',
              'alpha': 0.1,
              'num_iters': 50000,
              'prior_mu': 25,
              'prior_sigma': 3.0,
              'likelihood_sigma': 5.0,
              'gen_proc_mode1': 10,
              'gen_proc_mode2': 20,
              'gen_proc_std1': 3,
              'gen_proc_std2': 3,
              'gen_proc_weight1': 0.7,
              'num_obs': 5000,
              'mc_samples': 1000,
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
        mu, log_var = params[0], params[1]
        return mu, torch.exp(log_var)

    def log_q(samples, mu, sigma):
        log_q = -0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (samples - mu) **2 / sigma
        return log_q

    # Q samples
    mu, sigma = unpack_params(params)
    #print("unpacked params", mu, sigma)
    samples = (torch.randn(config['mc_samples']) * torch.sqrt(sigma) + mu)
    #print("samples", samples[0:10])

    #p0 = Normal(torch.tensor([config['prior_mu']],  dtype=torch.float32), torch.tensor([config['prior_sigma']],  dtype=torch.float32))
    mix = D.Categorical(torch.ones(2, ))
    comp = D.Normal(torch.tensor([11,19]), torch.tensor([1.0, 1.0]))
    p0 = D.MixtureSameFamily(mix, comp)


    factor = Normal(torch.tensor([obs],  dtype=torch.float32), torch.tensor([config['likelihood_sigma']],  dtype=torch.float32))

    logp0 = p0.log_prob(samples)
    logfactor = torch.mean(factor.log_prob(samples.reshape(-1, 1)), axis=1)

    # =============================================================================
    #         logp0 = log_prior(samples,1.0)
    # =============================================================================
    logq = log_q(samples, mu, sigma)

    if config['bound'] == 'Elbo':
        # =============================================================================
        #             lower_bound =  -(agnp.mean(gaussian_entropy(sigma) + (logp0+ logfactor)))
        # =============================================================================
        KL = torch.sum(-0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (mu ** 2 + sigma) / sigma) - \
             torch.sum(-0.5 * torch.log(2 * math.pi * sigma * np.exp(1)))
        lower_bound = -(torch.mean(logfactor) + KL)  # kl(mu, sigma))

    elif config['bound'] == 'Renyi':
        # print('ps',logp0 )
        # print('po',logfactor  )
        # print('q',logq  )
        logF = -logq + logp0 + logfactor
        # print('logF',logF)
        logF = (1 - config['alpha']) * logF
        # print('logF scaled',logF)
        lower_bound = torch.logsumexp(logF, 0) + torch.log(torch.as_tensor(1/config['mc_samples']))
        # print('Lower bound',lower_bound)
        lower_bound = lower_bound / (config['alpha']-1)
        # print('Lower bound scaled',lower_bound)


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

def generate_contour():

    means = np.arange(5.0, 25.0, 0.4)
    sigmas = np.arange(0.001, 5.0, 0.3)

    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for mean in means:
        print(mean)
        bounds = []
        for sigma in sigmas:
            params = torch.cat((torch.as_tensor([mean], dtype=torch.float32), torch.as_tensor([sigma], dtype=torch.float32)), 0).detach()
            torch.nn.utils.vector_to_parameters(params, policy.parameters())
            bound  , _, _ = compute_policy_loss(params)
            bounds.append(bound.detach().item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(means)*100,"%")
    #params[0].cpu().numpy().item()

    data.to_csv('/home/francesco/renyibounds/multi_armed_bandit/contour' + '.csv')
    return
#generate_contour()

def save_distributions():
    # plot p(s,o)
    samples = torch.arange(5.0, 25.0, 0.005)
    mix = D.Categorical(torch.ones(2, ))
    comp = D.Normal(torch.tensor([11, 19]), torch.tensor([1.0, 1.0]))
    p0 = D.MixtureSameFamily(mix, comp)
    factor = Normal(torch.tensor([obs], dtype=torch.float32),
                    torch.tensor([config['likelihood_sigma']], dtype=torch.float32))

    logp0 = p0.log_prob(samples)
    logfactor = torch.mean(factor.log_prob(samples.reshape(-1, 1)), axis=1)

    log_pso = logp0 + logfactor

    np.save('/home/francesco/renyibounds/multi_armed_bandit/samples',np.array(samples))

    po = torch.exp(logp0)
    np.save('/home/francesco/renyibounds/multi_armed_bandit/ps',np.array(po))

    pos = torch.exp(logfactor)
    np.save('/home/francesco/renyibounds/multi_armed_bandit/pos',np.array(pos))


    pso = torch.exp(log_pso)
    np.save('/home/francesco/renyibounds/multi_armed_bandit/pso',np.array(pso))


    samples2 = (torch.randn(config['mc_samples']) * torch.sqrt(torch.tensor(config['prior_sigma'])) + torch.tensor(config['prior_mu']))
    mix = D.Categorical(torch.ones(2, ))
    comp = D.Normal(torch.tensor([11, 19]), torch.tensor([1.0, 1.0]))
    p0 = D.MixtureSameFamily(mix, comp)
    factor = Normal(torch.tensor([obs], dtype=torch.float32),
                    torch.tensor([config['likelihood_sigma']], dtype=torch.float32))

    logp02 = p0.log_prob(samples2)
    logfactor2 = torch.mean(factor.log_prob(samples2.reshape(-1, 1)), axis=1)

    log_pso2 = logp02 + logfactor2
    pso2 = torch.exp(log_pso2)
    np.save('/home/francesco/renyibounds/multi_armed_bandit/pso2',np.array(pso2))
    np.save('/home/francesco/renyibounds/multi_armed_bandit/samples2',np.array(samples2))

    po2 = torch.exp(logp02)
    np.save('/home/francesco/renyibounds/multi_armed_bandit/ps2',np.array(po2))
    return