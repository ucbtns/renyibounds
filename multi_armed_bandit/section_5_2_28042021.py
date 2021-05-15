"""
@author: Noor S. 
@subject: Multi-armed bandits using Renyi Bounds  
"""

import os
os.chdir('D:\\Projects\\renyi_bound_analytics\\renyibounds\\multi_armed_bandit')
import utils_4 as ut 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm as norm
np.random.seed(1)
from scipy.special import softmax as softie
plot = False
run_simulation = True
import wandb 

congfig = {'alpha': '1', 'round': 100,
           'bou': 'a_1'}
               
               
wandb.init(project='renyibound', entity='vilmarith',config=congfig)
config = wandb.config

gmab = ut.MoGMAB([-20.0, 0, 0], [5,5, 5], [50,0,20],[5,5,5])      
fig = plt.figure(figsize=(7,4), dpi=1000)
X_plot = np.linspace(-50, 70, 1000)

weights_gmab= [[0.7,0.3], [1.0, 0.0], [.40,.60]]
bandit_colours = ['black', 'blue', 'orange']

if plot:
    
    for i in range(len(gmab.mu1)):   
        true_dens = (weights_gmab[i][0] * norm(gmab.mu1[i], gmab.sigma1[i]).pdf(X_plot)
                     + weights_gmab[i][1] * norm(gmab.mu2[i], gmab.sigma2[i]).pdf(X_plot))
        
        plt.fill(X_plot,true_dens, label='Arm {}'.format(i+1), alpha=0.5, color=bandit_colours[i])
    plt.legend()
    plt.xlabel('score'); plt.ylabel('density')
    plt.tight_layout();
    plt.show()


def simulator(n_rounds, policy):
    
    gmab = ut.MoGMAB([-20.0, 0, 0], [5,5, 5], [50,0,20],[5,10,5])

    n_bandits = len(gmab.mu1)
    k_list = []
    reward_list = []
    regret_list = []
    posterior_list = []
    arms = {'0': [], '1': [],'2': []}
    prior_mu = [50, 50, 50]
    regretsss= [-13, 0, 12]
    
    # loop generating draws
    for round_number in range(n_rounds):
        
        if policy == 'random':
            next_k_list = np.random.randint(0,3,10)
        elif policy == 'best':
            next_k_list = np.ones(1, np.int32)*2
        else:       
            next_k_list, posterior_estimates = policy.choose_bandit(k_list, reward_list, n_bandits, prior_mu)
            posterior_list.append(posterior_estimates)    
                 
            wandb.log({'Posterior_Arm 1': posterior_estimates[0][0]})
            wandb.log({'Posterior_Arm 2': posterior_estimates[1][0]})
            wandb.log({'Posterior_Arm 3': posterior_estimates[2][0]})
                
        
        for k in next_k_list:
            reward, _ = gmab.draw(k)
            prior_mu[k] = reward[0]
            regret = 12-regretsss[k] 
            k_list.append(k)
            reward_list.append(reward)
            regret_list.append(regret)
            arms[str(k)].append(reward)
        
            prior_mu[k] = (sum(arms[str(k)])/(len(arms[str(k)])))[0]
          
        wandb.log({'Regret': regret})
        wandb.log({'Reward': reward})
        wandb.log({'Arms Selected': next_k_list})
        
    # choices, rewards and regrets.
    return k_list, reward_list, regret_list, posterior_list


class VariationalPolicy:
    
    def __init__(self, bound, alpha,eta, iters): 
        self.bound = bound
        self.alpha = alpha
        self.iters = iters
        self.eta = eta
    
        
    def choose_bandit(self, k_list, reward_list, n_bandits, prior_mu):        
        k_list = np.array(k_list)
        reward_list = np.array(reward_list)

        bandit_post_samples = []
          
        for k in range(n_bandits):
            # filtering observation for this bandit
            obs = reward_list[k_list == k]
            infer = ut.variationalinference(prior_mu[k],1.0, 1.0, self.bound, self.alpha, 16)
            
            # performing inference and getting samples
            samples_post = infer.get_posterior(obs.T,self.eta,self.iters,k).rvs(1)          
            bandit_post_samples.append(samples_post)
           # wandb.log({'RB_Arm' + str(k):variational_objective})
                                
        return np.argmax(softie(bandit_post_samples),axis=0), bandit_post_samples
 

    
    
simulations =1
rounds = congfig['round']

simul_dict = {'a_inf': {'policy': VariationalPolicy('Max', 0, 0.1,10),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []},
              'a_1':   {'policy': VariationalPolicy('Elbo', 1,0.1,10),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []}, 
              'a_half': {'policy': VariationalPolicy('Renyi', 0.5,0.1,10),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []},
              'a_2': {'policy': VariationalPolicy('Renyi',2,0.1,10),
                        'regret': [],
                        'choices': [],
                         'rewards': [],
                         'posterior': []},
              'random': {'policy': 'random',
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []},
              'best': {'policy': 'best',
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []}}


if run_simulation:
    for bou in simul_dict.keys():
   #   if any([bou!='a_inf']): 
     if bou==congfig['bou']:
        for sim in tqdm(range(simulations)): 
            k_list, reward_list, regret_list,posterior_list = simulator(rounds, simul_dict[bou]['policy'])
    
            # storing results
            simul_dict[bou]['choices'].append(k_list)
            simul_dict[bou]['rewards'].append(reward_list)
            simul_dict[bou]['regret'].append(regret_list)
            simul_dict[bou]['posterior'].append(posterior_list) 
        simul_dict[bou]['policy'] = None
# =============================================================================
#           
#         with open('simulations_mab_27042021_'+bou+'.npy','wb') as ff:
#                         pickle.dump(simul_dict[bou],ff)    
# =============================================================================

