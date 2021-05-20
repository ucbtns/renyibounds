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
              'alpha': 0.5,
              'num_iters': 1000000,
              'prior_q': 25,
              'sigma_q': 1.0,
              'likelihood_sigma': 5.0,
              'prior_mu1': [8, 8, 8],
              'prior_sigma1': [3.0, 3.0, 3.0],
              'prior_mu2': [22,22, 22],
              'prior_sigma2': [3.0, 3.0, 3.0],
              'mixture_weight_prior': [0.5, 0.5, 0.5],
              'gen_proc_mode1': [10, 14, 14],
              'gen_proc_mode2': [25, 14, 19],
              'gen_proc_std1': [1, 1, 1],
              'gen_proc_std2': [1, 1, 1],
              'mixture_weight_gen_proc': [0.9, 1.0, 0.5],
              'num_obs': 5000,
              'mc_samples': 1000,
              }

wandb.init(project='renyi_0.5', config=def_config)
config = wandb.config
device = torch.device("cpu")
np.random.seed(1)


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

for arm in range(0, 3):
    policy.append(ut.VariationalPolicy(config['prior_q'], config['sigma_q']))
    optimize_policy.append(optim.Adam(policy[arm].parameters(), lr=config['learning_rate']))


def compute_log_prob(obs, params, mc_samples, likelihood_sigma, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2,
                     mixture_weight_prior, unif_samples=False):

    mu, log_var = params[0], params[1]
    sigma = torch.exp(log_var)

    if unif_samples:
        samples = torch.arange(5.0, 35.0, 0.005)
    else:
        samples = (torch.randn(mc_samples) * torch.sqrt(sigma) + mu)

    mix = D.Categorical(torch.tensor([mixture_weight_prior, 1-mixture_weight_prior]))
    comp = D.Normal(torch.tensor([prior_mu1, prior_mu2]), torch.tensor([prior_sigma1, prior_sigma2]))
    p0 = D.MixtureSameFamily(mix, comp)

    factor = Normal(torch.tensor([obs], dtype=torch.float32), torch.tensor([likelihood_sigma], dtype=torch.float32))

    logps = p0.log_prob(samples)
    logfactor = torch.mean(factor.log_prob(samples.reshape(-1, 1)), axis=1)
    logq = -0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (samples - mu) ** 2 / sigma

    return logps, logfactor, logq, samples


def learn(policy):
    obs = [[], [], []]
    rews = []
    arms_pulled = [0,0,0]
    for iter in range(config['num_iters']):


        arm = pull_arm(policy)
        rew = generate_obs(1, config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                                config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                                config['mixture_weight_gen_proc'][arm])[0]
        obs[arm].append(rew)
        rews.append(rew)
        arms_pulled[arm] += 1
         # Update policy
        params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)

        optimize_policy[arm].zero_grad()

        logps, logfactor, logq, _ = compute_log_prob(obs[arm], params, config['mc_samples'], config['likelihood_sigma'],
                                                  config['prior_mu1'][arm], config['prior_mu2'][arm], config['prior_sigma1'][arm],
                                                  config['prior_sigma2'][arm], config['mixture_weight_prior'][arm])

        loss_policy = ut.compute_policy_loss(config['mc_samples'], config['bound'], config['alpha'], logps, logfactor, logq)
        if iter % 200 == 0:
            #print("arm", arm, "iter", iter, "loss", loss_policy.detach().item(), "mu", params[0], "sigma", torch.exp(params[1]))
            p1 = nn.utils.parameters_to_vector(list(policy[0].parameters())).to(device, non_blocking=True)
            p2 = nn.utils.parameters_to_vector(list(policy[1].parameters())).to(device, non_blocking=True)
            p3 = nn.utils.parameters_to_vector(list(policy[2].parameters())).to(device, non_blocking=True)
            mu1 = p1[0].detach().item()
            mu2 = p2[0].detach().item()
            mu3 = p3[0].detach().item()


            print("iter", iter, "avg regret", 15.5 - np.mean(rews), "mus", [mu1,mu2,mu3], "frac_arm_pulled", arms_pulled)
            rews = []
            arms_pulled = [0,0,0]
        loss_policy.backward()
        optimize_policy[arm].step()


def pull_arm(policy):
    samples = []
    with torch.no_grad():
        for arm in range(0, 3):
            params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
            mu, log_var = params[0], params[1]
            sigma = torch.exp(log_var)

            samples.append((torch.randn(1) * torch.sqrt(sigma) + mu).item())
    arm_pulled = np.argmax(samples)
    #print(arm_pulled)

    return arm_pulled

#learn(policy)


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
    for arm in range(0, 3):

        params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
        obs = generate_obs(config['num_obs'], config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm], config['mixture_weight_gen_proc'][arm])

        logps, logfactor, logq, samples = compute_log_prob(obs, params, 1, config['likelihood_sigma'],
                                                  config['prior_mu1'][arm], config['prior_mu2'][arm],
                                                  config['prior_sigma1'][arm],
                                                  config['prior_sigma2'][arm], config['mixture_weight_prior'][arm], unif_samples=True)

        log_pso = logps + logfactor


        np.save('/home/francesco/renyibounds/multi_armed_bandit/samples_arm_'+str(arm),np.array(samples))

        ps = torch.exp(logps)
        np.save('/home/francesco/renyibounds/multi_armed_bandit/ps_arm_'+str(arm),np.array(ps))

        pos = torch.exp(logfactor)
        np.save('/home/francesco/renyibounds/multi_armed_bandit/pos_arm_'+str(arm),np.array(pos))

        pso = torch.exp(log_pso)
        np.save('/home/francesco/renyibounds/multi_armed_bandit/pso_arm_'+str(arm),np.array(pso))

        np.save('/home/francesco/renyibounds/multi_armed_bandit/rew_arm_'+str(arm),np.array(obs))

    return

#save_distributions()