# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:29:45 2021

@author: 44759
"""

import os
#os.chdir('D:\\Projects\\renyi_bound_analytics\\renyibounds\\multi_armed_bandit')
import utils as ut
import numpy as np
from scipy.stats import norm as norm_dist
import wandb
import torch.nn as nn
import torch.optim as optim
import torch
from torch.distributions.normal import Normal
import torch.distributions as D
import math

def_config = {'learning_rate': 1e-1,
              'bound': 'Renyi',
              'alpha': 2,
              'num_iters': 10000,
              'prior_q': 25,
              'sigma_q': 1e-8,
              'prior_mu1': [10, 10, 10],
              'prior_sigma1': [1.5, 1.5, 1.5],
              'prior_mu2': [17, 10, 17],
              'prior_sigma2': [1.5, 1.5, 1.5],
              'mixture_weight_prior': [0.5, 0.5, 0.5],
              'gen_proc_mode1': [10, 12, 12.5],
              'gen_proc_mode2': [20, 12, 14.5],
              'gen_proc_std1': [1, 2, 1],
              'gen_proc_std2': [1, 2, 1],
              'mixture_weight_gen_proc': [0.98, 1.0, 0.5],
              'num_obs': 1000,
              'mc_samples': 300,
              'save_distributions': True,
              'train': True,
              'run': 1,
              'log_every': 100,
              'num_updates': 10,
              'log_reg_every': 100,
              'print_log': True,
              'generate_contour': False,
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

oracle = []
for arm in range(0, 3):
    obs = generate_obs(1000, config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                       config['mixture_weight_gen_proc'][arm])
    oracle.append(np.mean(obs)/np.var(obs))
print("sharpe:", oracle)



policy = []
optimize_policy = []

# Initialize variational posterior and optimizer
for arm in range(0, 3):
    policy.append(ut.VariationalPolicy(config['prior_q'], config['sigma_q']))
    optimize_policy.append(optim.Adam(policy[arm].parameters(), lr=config['learning_rate']))


def compute_log_prob(obs, params, mc_samples, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2,
                     mixture_weight_prior, unif_samples=False):

    mu, log_var = params[0], params[1]
    sigma = torch.exp(log_var)

    if unif_samples:
        samples = torch.arange(0.0, 30.0, 0.01)
    else:
        samples = (torch.randn(mc_samples) * torch.sqrt(sigma) + mu)

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


post_sharpe = []
for arm in range(0, 3):
    params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
    obs = generate_obs(config['num_obs'], config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                       config['mixture_weight_gen_proc'][arm])

    logps, logfactor, logq, samples = compute_log_prob(obs, params, 1,
                                                       config['prior_mu1'][arm], config['prior_mu2'][arm],
                                                       config['prior_sigma1'][arm],
                                                       config['prior_sigma2'][arm], config['mixture_weight_prior'][arm],
                                                       unif_samples=True)

    log_pso = logps + logfactor

    int = torch.mean(torch.exp(log_pso))

    pso = torch.exp(log_pso)


    posterior = np.multiply(np.exp(log_pso), 1 / int)

    mean_from_samples = torch.mean(samples * posterior)
    var_from_samples = torch.mean((samples - mean_from_samples) ** 2 * posterior)

    print("arm", arm, "post mean", mean_from_samples, "post var", var_from_samples, "post sharpe",
          mean_from_samples / var_from_samples)

    post_sharpe.append(mean_from_samples / var_from_samples)

print("post_sharpe", post_sharpe)


def learn(policy):
    obs = [[], [], []]
    rews = []
    arms_pulled = [0,0,0]
    for iter in range(config['num_iters']):

        # Sample each arm once at the beginning
        if iter == 0 or iter == 1:
            arm = 0
        elif iter == 2 or iter == 3:
                arm = 1
        elif iter == 4 or iter == 5:
            arm = 2
        else:
            arm = pull_arm(policy)

        #print(arm)
        rew = generate_obs(1, config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                                config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                                config['mixture_weight_gen_proc'][arm])[0]
        obs[arm].append(rew)

        # Keep a fixed amount of observations
        if len(obs[arm]) > config['num_obs']:
            del obs[arm][0]

        rews.append(rew)
        arms_pulled[arm] += 1

        # Learn
        if iter > 5:

            for _ in range(config['num_updates']):

                params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)

                optimize_policy[arm].zero_grad()

                logps, logfactor, logq, samples = compute_log_prob(obs[arm], params, config['mc_samples'],
                                                          config['prior_mu1'][arm], config['prior_mu2'][arm], config['prior_sigma1'][arm],
                                                          config['prior_sigma2'][arm], config['mixture_weight_prior'][arm])

                loss_policy = ut.compute_policy_loss(config['mc_samples'], config['bound'], config['alpha'], logps, logfactor, logq)
                loss_policy.backward()
                optimize_policy[arm].step()

        # Log results
        if iter % config['log_every'] == 0:
            p1 = nn.utils.parameters_to_vector(list(policy[0].parameters())).to(device, non_blocking=True)
            p2 = nn.utils.parameters_to_vector(list(policy[1].parameters())).to(device, non_blocking=True)
            p3 = nn.utils.parameters_to_vector(list(policy[2].parameters())).to(device, non_blocking=True)
            mu1 = p1[0].detach().item()
            mu2 = p2[0].detach().item()
            mu3 = p3[0].detach().item()
            sigma1 = torch.exp(p1[1]).detach().item()
            sigma2 = torch.exp(p2[1]).detach().item()
            sigma3 = torch.exp(p3[1]).detach().item()

            def evaluate_bound():
                bounds = []

                for arm in range(0, 3):
                    with torch.no_grad():
                        params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)

                        logps, logfactor, logq, samples = compute_log_prob(obs[arm], params, config['mc_samples'],
                                                                           config['prior_mu1'][arm],
                                                                           config['prior_mu2'][arm],
                                                                           config['prior_sigma1'][arm],
                                                                           config['prior_sigma2'][arm],
                                                                           config['mixture_weight_prior'][arm])
                        bound = ut.compute_policy_loss(config['mc_samples'], config['bound'], config['alpha'], logps,
                                                             logfactor, logq)
                        bounds.append(bound.detach().item())


                return bounds

            bounds = evaluate_bound()
            if config['print_log']:
                if iter > 2:

                    print("iter", iter, "bounds", bounds, "avg regret",
                          np.max(post_sharpe)-(np.mean(rews)/np.var(rews)),
                          "mus", [mu1,mu2,mu3], "frac_arm_pulled", arms_pulled,
                        "Sigmas", [sigma1, sigma2, sigma3])


            wandb.log({'mu1': mu1,
                       'mu2': mu2,
                       'mu3': mu3,
                       'sigma1': sigma1,
                       'sigma2': sigma2,
                       'sigma3': sigma3,
                       'bound1': bounds[0],
                       'bound2': bounds[1],
                       'bound3': bounds[2],
                       'iter': iter,
                       })


        if iter % 100 == 0:

            frac_arm1 = arms_pulled[0]/100
            frac_arm2 = arms_pulled[1]/100
            frac_arm3 = arms_pulled[2]/100

            wandb.log({'frac_arm1': frac_arm1,
                   'frac_arm2': frac_arm2,
                   'frac_arm3': frac_arm3,
                   'iter2': iter})

            arms_pulled = [0,0,0]

        if iter % config['log_reg_every'] == 0:
            if iter > 2:
                wandb.log({'avg_regret': 13.5 - np.mean(rews), 'iter3': iter})

                rews = []


def pull_arm(policy):
    samples = []
    with torch.no_grad():
        for arm in range(0, 3):
            params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
            mu, log_var = params[0], params[1]
            sigma = torch.exp(log_var)

            samples.append(((torch.randn(1) * torch.sqrt(sigma) + mu)/sigma).item())
    arm_pulled = np.argmax(samples)
    return arm_pulled


# Train
if config['train']:
    learn(policy)


def generate_contour():
    arm=0
    means = np.arange(5.0, 25.0, 0.2)
    sigmas = np.arange(0.001, 5.0, 0.2)
    obs = generate_obs(1000, config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm],
                       config['mixture_weight_gen_proc'][arm])
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for mean in means:
        bounds = []
        for sigma in sigmas:
            params = torch.cat((torch.as_tensor([mean], dtype=torch.float32), torch.as_tensor([sigma], dtype=torch.float32)), 0).detach()

            logps, logfactor, logq, samples = compute_log_prob(obs, params, config['mc_samples'],
                                                               config['prior_mu1'][arm], config['prior_mu2'][arm],
                                                               config['prior_sigma1'][arm],
                                                               config['prior_sigma2'][arm],
                                                               config['mixture_weight_prior'][arm], unif_samples=True)

            bound = ut.compute_policy_loss(300, config['bound'], config['alpha'], logps,
                                                 logfactor, logq)

            bounds.append(bound.detach().item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(means)*100,"%")

    data.to_csv(dirname+'/contour' + '.csv')
    return


# Print optimization landscape for arm 1
if config['generate_contour']:
    generate_contour()


def save_distributions():
    for arm in range(0, 3):

        params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(device, non_blocking=True)
        obs = generate_obs(config['num_obs'], config['gen_proc_mode1'][arm], config['gen_proc_std1'][arm],
                       config['gen_proc_mode2'][arm], config['gen_proc_std2'][arm], config['mixture_weight_gen_proc'][arm])

        logps, logfactor, logq, samples = compute_log_prob(obs, params, 1,
                                                  config['prior_mu1'][arm], config['prior_mu2'][arm],
                                                  config['prior_sigma1'][arm],
                                                  config['prior_sigma2'][arm], config['mixture_weight_prior'][arm], unif_samples=True)

        log_pso = logps + logfactor

        int = torch.mean(torch.exp(log_pso))
        print("integral of p(s,o) for arm ", arm, ":", int)
        print("mean of arm", arm, "is", np.mean(obs), "var", np.var(obs), 'sharpe', np.mean(obs)/np.var(obs))

        np.save(dirname+'/samples_arm_'+str(arm),np.array(samples))

        ps = torch.exp(logps)
        np.save(dirname+'/ps_arm_'+str(arm),np.array(ps))

        pos = torch.exp(logfactor)
        np.save(dirname+'/pos_arm_'+str(arm),np.array(pos))

        pso = torch.exp(log_pso)
        np.save(dirname+'/pso_arm_'+str(arm),np.array(pso))

        np.save(dirname+'/rew_arm_'+str(arm),np.array(obs))

        posterior = np.multiply(np.exp(log_pso), 1/int)

        mean_from_samples = torch.mean(samples * posterior)
        var_from_samples = torch.mean((samples-mean_from_samples)**2 * posterior)

        print("arm", arm, "post mean", mean_from_samples, "post var", var_from_samples, "post sharpe", mean_from_samples/var_from_samples)
    return


# Save logp for each arm
if config['save_distributions']:
    save_distributions()

