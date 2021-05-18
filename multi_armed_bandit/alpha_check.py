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
np.random.seed(1)
import autograd.numpy as agnp

from scipy.stats import norm as norm_dist
plot = False
from autograd.misc.optimizers import adam
run_simulation = True
import wandb 



def simulator(policy, obs, bound, alpha, prior_mu, prior_sigma, likelihood_sigma, mc_samples):
    
    infer = ut.Variationalinference(prior_mu, prior_sigma, likelihood_sigma, bound, alpha, mc_samples)
    init_param1, gradient = infer.get_posterior(obs)

    policy.test_post(gradient)

class VariationalPolicy:

    def __init__(self, bound, alpha, step_size, num_iters, prior_mu, prior_sigma):
        self.bound = bound
        self.alpha = alpha
        self.num_iters = num_iters
        self.step_size = step_size
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
    def test_post(self, gradient):

        init_params =  agnp.array([self.prior_mu, self.prior_sigma])
        variational_params = adam(gradient, init_params, step_size=self.step_size, num_iters=self.num_iters)
        wandb.log({'Posterior mu': variational_params[0]})
        wandb.log({'Posterior std': np.exp(variational_params[1])})


#for i in  range(1,100,2):
      #  i = i/1000

def_config = {'round': 1000,
              'bou': 'reny_bms',
              'learning_rate': 5e-1,
              'bound': 'Renyi',
              'alpha': 10,
              'num_iters': 3,
              'simulations': 1,
              'prior_mu': 50,
              'prior_sigma': 1.0,
              'likelihood_sigma': 1.0,
              'gen_proc_mode1': 0,
              'gen_proc_mode2': 20,
              'gen_proc_std1': 5,
              'gen_proc_std2': 5,
              'gen_proc_weight1': 0.7,
              'num_obs': 5000,
              'mc_samples': 5,
              'run': 1,
              }

wandb.init(project='renyi_0.5', config=def_config)
config = wandb.config

gmab = ut.MoGMAB(config['gen_proc_mode1'], config['gen_proc_std1'], config['gen_proc_mode2'],
                 config['gen_proc_std2'], config['gen_proc_weight1'])

obs = []
for j in range(config['num_obs']):
    reward = gmab.draw()
    obs.append(reward)

#import matplotlib.pyplot as plt
#fig = plt.hist(obs)
#plt.show()


#np.save('obs.npy', obs)

policy = VariationalPolicy(config['bound'], config['alpha'], config['learning_rate'], config['num_iters'],
                           config['prior_mu'], config['prior_sigma'])


for sim in tqdm(range(config['simulations'])):
     simulator(policy, obs, config['bound'], config['alpha'], config['prior_mu'],
               config['prior_sigma'], config['likelihood_sigma'], config['mc_samples'])
