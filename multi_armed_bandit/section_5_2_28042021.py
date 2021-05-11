"""
@author: Noor S. 
@subject: Multi-armed bandits using Renyi Bounds  
"""

import os
os.chdir('D:/PhD/Draft Papers/Current/renyi/')
import utils_4 as ut 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm as norm
np.random.seed(1)
import pandas as pd
from scipy.special import softmax as softie
plot = False
run_simulation = False

gmab = ut.MoGMAB([-20.0, 2, 0], [5,5, 5], [50,0,20],[5,10,5])      
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
    
    gmab = ut.MoGMAB([-20.0, 2, 0], [5,5, 5], [50,0,20],[5,10,5])

    n_bandits = len(gmab.mu1)
    k_list = []
    reward_list = []
    regret_list = []
    posterior_list = []

    # loop generating draws
    for round_number in range(n_rounds):
        
        if policy == 'random':
            next_k_list = np.random.randint(0,3,10)
        elif policy == 'best':
            next_k_list = np.ones(10, np.int32)*2
        else:       
            next_k_list, posterior_estimates = policy.choose_bandit(k_list, reward_list, n_bandits)
            posterior_list.append(posterior_estimates)          
     
        for k in next_k_list:
            reward, regret = gmab.draw(k)
            k_list.append(k)
            reward_list.append(reward)
            regret_list.append(regret)

    # choices, rewards and regrets.
    return k_list, reward_list, regret_list, posterior_list


class VariationalPolicy:
    
    def __init__(self, bound, alpha,eta, iters): 
        self.bound = bound
        self.alpha = alpha
        self.iters = iters
        self.eta = eta
        
    def choose_bandit(self, k_list, reward_list, n_bandits):        
        k_list = np.array(k_list)
        reward_list = np.array(reward_list)
    
        infer = ut.variationalinference([0.0, 1.0], [1.0, 1.0], 1, [0.5,0.5], self.bound, self.alpha, 16)
        bandit_post_samples = []
          
        for k in range(n_bandits):
            # filtering observation for this bandit
            obs = reward_list[k_list == k]
            
            # performing inference and getting samples
            # 1500, 750 , > 150 
            samples_post = infer.get_posterior(obs.T,self.eta,self.iters).rvs(10)
            bandit_post_samples.append(samples_post)
                                
        return np.argmax(softie(bandit_post_samples),axis=0), bandit_post_samples
    
       
simulations =20
rounds = 100

simul_dict = {'a_inf': {'policy': VariationalPolicy('Max', 0, 0.04,5000),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []},
              'a_1':   {'policy': VariationalPolicy('Elbo', 1,0.01,256),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []}, 
              'a_half': {'policy': VariationalPolicy('Renyi', 0.5,0.02,2048),
                        'regret': [],
                        'choices': [],
                        'rewards': [],
                         'posterior': []},
              'a_2': {'policy': VariationalPolicy('Renyi',2,0.01,512),
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


import pickle 
  
if run_simulation:
    for bou in simul_dict.keys():
      if any([bou!='a_inf']): 
        for sim in tqdm(range(simulations)): 
            k_list, reward_list, regret_list,posterior_list = simulator(rounds, simul_dict[bou]['policy'])
    
            # storing results
            simul_dict[bou]['choices'].append(k_list)
            simul_dict[bou]['rewards'].append(reward_list)
            simul_dict[bou]['regret'].append(regret_list)
            simul_dict[bou]['posterior'].append(posterior_list) 
        simul_dict[bou]['policy'] = None
          
        with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_'+bou+'.npy','wb') as ff:
                        pickle.dump(simul_dict[bou],ff)    


# Loading the data:
with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_a_half.npy', 'rb') as f: 
    simul_dict['a_half'] = pickle.load(f)
with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_a_1.npy', 'rb') as f: 
    simul_dict['a_1'] = pickle.load(f)
        
with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_a_2.npy', 'rb') as f: 
    simul_dict['a_2'] = pickle.load(f)
with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_a_inf.npy', 'rb') as f: 
    simul_dict['a_inf'] = pickle.load(f)
      
    
with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_random.npy', 'rb') as f: 
    simul_dict['random'] = pickle.load(f)

with open('D:/PhD/Draft Papers/Current/renyi/data/simulations_mab_27042021_best.npy', 'rb') as f: 
    simul_dict['best'] = pickle.load(f)

# Plotting:
# ==========================================
alpha = [r'$\alpha \to 0.5$', r'$\alpha= 1$', r'$\alpha \to 2$', r'$\alpha \to \infty$', 'Random', 'Best Policy']
lines = [2]*6
  
fig = plt.figure(figsize=(10,7), dpi=1000)
num = 0
for bou in simul_dict.keys(): 

        dat = np.zeros((1000,20))
        da  = simul_dict[bou]['rewards']
        for i in range(0,simulations):
           # if (bou == 'a_neginf'):
            #    print(bou)
             #   dat[:,i] = np.reshape(da[i+20], (1000))
            #else:
                dat[:,i] = np.reshape(da[i], (1000))
                
    
        avg_cum_regret = np.cumsum(dat, axis=0)
        means = np.mean(avg_cum_regret, axis=1)
        stds = np.std(avg_cum_regret, axis=1)
        plt.plot(means, label=alpha[num], linewidth=lines[num])
        plt.fill_between(range(0,1000), means+ stds, means-stds, alpha=0.1)
        plt.xlim(0,1000);
        num += 1

#plt.legend()
plt.legend(loc='best', fontsize=15)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel('Round', fontsize=15); plt.ylabel('Cumulative Score', fontsize=15);

countsDAT = np.zeros((6*20*3,3))

n = 0
for num, bou in enumerate(simul_dict.keys()):
    da  = simul_dict[bou]['choices']
    counts = np.zeros((20*3,3))

    for i in range(0,simulations):
        #if (bou == 'a_neginf'):
         #   dat[:,i] = np.reshape(da[i+20], (1000))
        #else:
        dat[:,i] = np.reshape(da[i], (1000))   
        u,c = np.unique(dat[:,i], return_counts=True) 
        try:
                counts[(i*3):3+(i*3),1] = c 
                counts[(i*3):3+(i*3),0] = u
        except:
                counts[(1+i*3):3+(i*3),1] = c 
                counts[(i*3):3+(i*3),0] = [0,1,2]
                     
    counts[:,2] = num
    countsDAT[n:n+20*3,:]  = counts
    n += 60     
    

uniques = pd.DataFrame(countsDAT)
uniques.columns =['Arm', 'Count', r'$\alpha$']
uniques[r'$\alpha$'] = uniques[r'$\alpha$'].replace({0.0: alpha[0], 1.0: alpha[1], 2.0: alpha[2],3.0: alpha[3], 4.0: alpha[4], 5.0: alpha[5]})
uniques[' '] = uniques['Arm'].replace({0.0: 'Arm 1', 1.0: 'Arm 2', 2.0: 'Arm 3'})


import seaborn as sns

plt.figure(figsize=(15,5), dpi=1000)
sns.set_style("ticks")
#sns.set(font_scale = 1.2)
sns.catplot(kind='bar', hue=' ', y="Count",
                x= r'$\alpha$',
                data=uniques, palette=bandit_colours,alpha=0.6);
plt.ylabel("Arm selected (count)",fontsize=15); plt.xlabel(r'$\alpha$ value',fontsize=15)


poster_data= np.zeros((1000,3*20,4))
nn = 0
for bou in simul_dict.keys(): 
    if not (bou == 'random' or bou == 'best'):
            da  = simul_dict[bou]['posterior']
            for j in range(0,simulations):
                for i in range(0,rounds):
                    #if (bou == 'a_neginf'):
                    #    poster_data[(i*10):10+(i*10),[j,20+j,40+j],nn] = np.transpose(da[j+20][i])
                    #else:
                        poster_data[(i*10):10+(i*10),[j,20+j,40+j],nn] = np.transpose(da[j][i])


            nn += 1
         
plt.figure(figsize=(10,5), dpi=1000)
plt.subplot(1,3, 1)
nn = 0       
while nn <4:
    means_0 = np.median(poster_data[:,0:20,nn], axis=1)
    plt.plot(means_0, label = alpha[nn])
    #plt.fill_between(range(0,1000), means_0+ stds_0, means_0-stds_0, alpha=0.1)
    plt.xlim(0,1000);
    plt.ylim(-5,10);
    plt.xlabel('Arm 1')
    plt.ylabel(r'$q(s_{1})$')
    plt.legend(loc='upper right', fontsize=9) 
    plt.tight_layout()
    nn +=1
    
plt.subplot(1, 3, 2)
nn = 0      
while nn <4:
    means_1 = np.median(poster_data[:,20:40,nn], axis=1)
    plt.plot(means_1, label = alpha[nn])
    #plt.fill_between(range(0,1000), means_0+ stds_0, means_0-stds_0, alpha=0.1)
    plt.xlim(0,1000);
    plt.ylim(-5,10);
    #plt.legend(loc='lower right', fontsize=9) 
    plt.xlabel('Arm 2')
    plt.ylabel(r'$q(s_{2})$')
    plt.tight_layout()
    nn +=1

plt.subplot(1,3, 3)
nn = 0       
while nn <4:
    means_2 = np.median(poster_data[:,40:60,nn], axis=1)
    plt.plot(means_2, label = alpha[nn])
    #plt.fill_between(range(0,1000), means_0+ stds_0, means_0-stds_0, alpha=0.1)
    plt.xlim(0,1000);
    plt.ylim(-5,10);
    plt.xlabel('Arm 3')
    plt.ylabel(r'$q(s_{3})$')
    #plt.legend(loc='lower right', fontsize=9) 
    nn +=1
    plt.tight_layout()

    

