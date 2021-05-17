"""
@author: Noor S. 
@subject: Multi-armed bandits using Renyi Bounds  
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



gmab = ut.MoGMAB([-20.0, 0, 0], [5,5, 5], [50,0,20],[5,10,5])
weights_gmab= [[0.7,0.3], [1.0, 0.0], [.40,.60]]

obs = {'0': [], '1': [],'2': []}
for j in range(1000):
    for k in range(3):
        reward, _ = gmab.draw(k)
        obs[str(k)].append(reward[0])
            
np.save('obs.npy', obs)  

# ============================================================================= 
# =============================================================================


def simulator(policy, obs, gmab, bound, alpha):
    
    n_bandits = len(gmab.mu1)
    infer1 = ut.variationalinference(50,1.0, 1.0, bound, alpha, 16)
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
        regretsss= [-13, 0, 12]
        
        for j in range(60):  
            bandit_post_samples = []  
            for k in range(n_bandits):
                gradient = posts[str(k)]
                    
                if j == 0:                 
                    init_params[str(k)]  =  agnp.array([50.0, 1.0])
                    variational_params = adam(gradient, init_params[str(k)] , step_size=1e-100, num_iters=1)
                    samples = norm_dist(variational_params[0], variational_params[1])
                else: 
                    variational_params = adam(gradient, init_params[str(k)] , step_size=self.eta, num_iters=32)
                    init_params[str(k)] =  agnp.array([variational_params[0], variational_params[1]])
                
                wandb.log({'Posterior mu' + str(k): variational_params[0]})
                wandb.log({'Posterior std' + str(k): np.exp(variational_params[1])})
                samples = norm_dist(variational_params[0], np.exp(variational_params[1]))
                samples_post = np.mean(samples.rvs(10))
                bandit_post_samples.append(np.array([samples_post]))    
                
                if k == 2:
                    arm_selected = np.argmax(bandit_post_samples,axis=0)[0]
                    reward, _ = gmab.draw(arm_selected)
                    regret = 12-regretsss[arm_selected] 
            
                    wandb.log({'Regret': regret})
                    wandb.log({'Reward': reward})
                    wandb.log({'Arms Selected': arm_selected})
                    
       
# =============================================================================
# =============================================================================

bounds = ['a_inf','a_1', 'a_half','a_2', 'a_0_1']
#st = [1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 5e-2, 1e-2,5e-1, 1e-1]# [2, 1.5, 1.1, 1.3] #
bounds = ['negMax']
st =  [1e-5, 5e-5, 1e-4, 5e-4, 5e-3, 1e-3, 5e-2, 1e-2,5e-1, 1e-1, 5e-1, 1e-1, 4e-1, 3e-1, 2e-1, 6e-1, 8e-1, 9e-1, 1 , 5e-1]
for bou in bounds:
        
    for steps in st: 
        if bou == 'a_inf': policy = VariationalPolicy('Max', 10000, steps,100); bound = 'Max'; alpha = 100000
        elif bou == 'a_1': policy= VariationalPolicy('Elbo', 1,steps,100); bound = 'Elbo'; alpha = 1
        elif bou == 'a_half': policy = VariationalPolicy('Renyi', 0.5, steps,100); bound = 'Renyi'; alpha = 0.5
        elif bou == 'a_2': policy= VariationalPolicy('Renyi', 2,steps,100); bound = 'Renyi'; alpha = 2
        elif bou == 'a_0_1': policy= VariationalPolicy('Renyi', 0.01,steps,100); bound = 'Renyi'; alpha = 0.01
        elif bou == 'negMax':  policy= VariationalPolicy('negMax', -10000,steps,100); bound = 'negMax'; alpha = -100000
        
        
        congfig = {'alpha': alpha, 'round': 1000,
               'bou': bou, 'learning_rate': steps}
                                  
        wandb.init(project='reny_neg', entity='vilmarith',config=congfig)
        config = wandb.config
        simulations =1
        rounds = congfig['round']
        for sim in tqdm(range(simulations)): 
             simulator(policy, obs, gmab, bound, alpha)
             
    
              
