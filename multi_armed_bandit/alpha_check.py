# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:29:45 2021

@author: 44759
"""

import os
os.chdir('D:\\Projects\\renyi_bound_analytics\\renyibounds\\multi_armed_bandit')
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



gmab = ut.MoGMAB([0, 0, 0], [5,5, 5], [100,0,20],[5,10,5])
weights_gmab= [[0.7,0.3], [1.0, 0.0], [.40,.60]]

obs = {'0': [], '1': [],'2': []}
for j in range(5000):
    for k in range(3):
        reward, _ = gmab.draw(k)
        obs[str(k)].append(reward[0])
            
np.save('obs.npy', obs)  

# ============================================================================= 
# =============================================================================


def simulator(policy, obs, gmab, bound, alpha):
    
    n_bandits = len(gmab.mu1)
    infer1 = ut.variationalinference(50,1.0, 1.0, bound, alpha, 1)
    infer2 = ut.variationalinference(50,1.0, 1.0, bound, alpha, 16)
    infer3 = ut.variationalinference(50,1.0, 1.0, bound, alpha, 16)
    init_param1, gradient1 = infer1.get_posterior(obs['0'],0.1,100,0) 
    init_param2, gradient2 = infer2.get_posterior(obs['1'],0.1,100,1) 
    init_param3, gradient3 = infer3.get_posterior(obs['2'],0.1,100,2) 
     
    grads = {'0': gradient1, '1': gradient2,'2': gradient3}
    policy.test_post( n_bandits,  grads)                



class VariationalPolicy:
    
    def __init__(self, bound, alpha,eta, iters): 
        self.bound = bound
        self.alpha = alpha
        self.iters = iters
        self.eta = eta
            
    def test_post(self,  n_bandits, posts):        
       
        init_params = {'0':  agnp.array([50, 1]), '1':  agnp.array([50, 1]),'2':  agnp.array([50, 1])}  
                
        k = 0
        gradient = posts[str(k)]
                    
        init_params[str(k)]  =  agnp.array([50.0, 1.0])
        variational_params = adam(gradient, init_params[str(k)] , step_size=self.eta, num_iters=32)       
        wandb.log({'Posterior mu' + str(k): variational_params[0]})
        wandb.log({'Posterior std' + str(k): np.exp(variational_params[1])})
                
                    
       
# =============================================================================
# =============================================================================

steps= 5e-1
     
for i in  range(1,100,2):
      #  i = i/1000
        bound = 'Renyi'; 
        alpha = -10
        
        policy= VariationalPolicy(bound, alpha,steps,100); 

        
        congfig = {'alpha': alpha, 'round': 1000,
               'bou': 'reny_bms', 'learning_rate': steps}
                                  
        wandb.init(project='renyi_0.5', entity='vilmarith',config=congfig)
        config = wandb.config
        simulations =1
        rounds = congfig['round']
        for sim in tqdm(range(simulations)): 
             simulator(policy, obs, gmab, bound, alpha)
             
    
              