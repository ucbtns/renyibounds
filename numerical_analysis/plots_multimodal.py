# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:29:45 2021

@author: 44759
"""

import os
from multi_armed_bandit import utils as ut
import numpy as np
import wandb
import torch.optim as optim
import torch
from torch.distributions.normal import Normal
import torch.distributions as D
import math
import torch.nn as nn

def_config = {'learning_rate': 1e-1,
              'bound': 'Renyi',
              'alpha': 1e9,
              'num_iters': 10000,
              'prior_q': 25,
              'sigma_q': 1.0,
              'prior_mu1': [10, 10, 10],
              'prior_sigma1': [1.5, 1.5, 1.5],
              'prior_mu2': [17, 17, 17],
              'prior_sigma2': [1.5, 1.5, 1.5],
              'mixture_weight_prior': [0.5, 0.5, 0.5],
              'gen_proc_mode1': [10, 12, 12.5],
              'gen_proc_mode2': [20, 12, 14.5],
              'gen_proc_std1': [1, 0.5, 0.3],
              'gen_proc_std2': [1, 0.5, 0.3],
              'mixture_weight_gen_proc': [0.8, 1.0, 0.5],
              'num_obs': 1000,
              'mc_samples': 1000,
              'save_distributions': True,
              'train': False,
              'run': 1,
              'log_every': 100,
              'num_updates': 10,
              'log_reg_every': 100,
              'print_log': True,
              'generate_contour': True,
              }

dirname = os.path.dirname(__file__)
wandb.init(project='renyi_0.5', config=def_config)
config = wandb.config
device = torch.device("cpu")


def generate_obs(num_obs, gen_proc_mode1, gen_proc_std1, gen_proc_mode2, gen_proc_std2, gen_proc_weight1):
    obs = []
    for j in range(num_obs):
        y1 = np.random.normal(gen_proc_mode1, gen_proc_std1)
        y2 = np.random.normal(gen_proc_mode2, gen_proc_std2)
        w = np.random.binomial(1, gen_proc_weight1, 1)
        reward = w*y1 + (1-w)*y2
        obs.append(reward.item())
    return obs


policy = []
optimize_policy = []

# Initialize variational posterior and optimizer
for arm in range(0, 3):
    policy.append(ut.VariationalPolicy(config['prior_q'], config['sigma_q']))
    optimize_policy.append(optim.Adam(policy[arm].parameters(), lr=config['learning_rate']))


def compute_log_prob(samples, obs, params, mc_samples, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2,
                     mixture_weight_prior, unif_samples=False):

    mu, log_var = params[0], params[1]
    sigma = torch.exp(log_var)



    mix = D.Categorical(torch.tensor([mixture_weight_prior, 1-mixture_weight_prior]))
    comp = D.Normal(torch.tensor([prior_mu1, prior_mu2]), torch.tensor([prior_sigma1, prior_sigma2]))
    p0 = D.MixtureSameFamily(mix, comp)
    std_factor = torch.std(torch.tensor([obs], dtype=torch.float32))
    var_factor = std_factor ** 2
    factor = Normal(torch.tensor([obs], dtype=torch.float32), var_factor)

    logps = p0.log_prob(samples)
    logfactor = torch.mean(factor.log_prob(samples.reshape(-1, 1)), axis=1)
    logq = -0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (samples - mu) ** 2 / sigma

    return logps, logfactor, logq, samples



def save_distributions():
    arm=0

    params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
    obs = generate_obs(config['num_obs'], config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                   config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm], config['mixture_weight_gen_proc'][arm])

    #samples_unscaled = torch.randn(config['mc_samples'])
    samples = torch.arange(0.0, 30.0, 0.003)

    logps, logfactor, logq, samples = compute_log_prob(samples, obs, params, 1,
                                              config['prior_mu1'][arm], config['prior_mu2'][arm],
                                              config['prior_sigma1'][arm],
                                              config['prior_sigma2'][arm], config['mixture_weight_prior'][arm], unif_samples=True)

    log_pso = logps + logfactor

    int = torch.mean(torch.exp(log_pso)) * 30
    print("integral of p(s,o) for arm ", arm, ":", int)

    np.save(dirname+'/multimodal_samples',np.array(samples))

    ps = torch.exp(logps)
    np.save(dirname+'/multimodal_ps',np.array(ps))

    pos = torch.exp(logfactor)
    np.save(dirname+'/multimodal_pos',np.array(pos))

    pso = torch.exp(log_pso)
    np.save(dirname+'/multimodal_pso',np.array(pso))

    np.save(dirname+'/multimodal_rew',np.array(obs))
    return


# Save logp for each arm
if config['save_distributions']:
    save_distributions()





def generate_contour():
    alphas = [1e-6, 0.5, 0.99, 1.01, 2, 1e6]

    arm=0
    means = np.arange(0.0, 26.0, 0.2)
    sigmas = np.arange(0.001, 5.0, 0.2)
    obs = generate_obs(config['num_obs'], config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                       config['mixture_weight_gen_proc'][arm])
    import pandas as pd


    i = 0
    for alpha in alphas:
        data = pd.DataFrame()
        samples_unscaled = torch.randn(config['mc_samples'])
        for mean in means:
            bounds = []
            for sigma in sigmas:
                params = torch.cat((torch.as_tensor([mean], dtype=torch.float32), torch.as_tensor([sigma], dtype=torch.float32)), 0).detach()

               # if unif_samples:
               #     samples = torch.arange(0.0, 30.0, 0.01)
               # else:
                mu_q, log_var = params[0], params[1]
                sigma_q = torch.exp(log_var)
                samples = (samples_unscaled * torch.sqrt(sigma_q) + mu_q)

                logps, logfactor, logq, samples = compute_log_prob(samples, obs, params, config['mc_samples'],
                                                                   config['prior_mu1'][arm], config['prior_mu2'][arm],
                                                                   config['prior_sigma1'][arm],
                                                                   config['prior_sigma2'][arm],
                                                                   config['mixture_weight_prior'][arm], unif_samples=False)

                bound = ut.compute_policy_loss(config['mc_samples'], config['bound'], alpha, logps,
                                                     logfactor, logq)

                bounds.append(bound.detach().item())

            data['bound_'+str(i)] = bounds
            i += 1
            print("alpha", alpha, i/len(means)*100,"%")

        data.to_csv(dirname+'/contour_'+str(alpha) + '.csv')
    return


# Print optimization landscape for arm 1
if config['generate_contour']:
    generate_contour()

