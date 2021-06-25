
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
dirname = 'D:\\Projects\\renyi_bound_analytics\\renyibounds\\data\\'
os.chdir(dirname)

samples0 = list(np.load(dirname+'\\samples_arm_0.npy'))

muinf = 16.27
varinf = 2

mu10 = 10.3
var10= 1.5

mu2 = 10.2
var2 = 1.8

mu1 = 10.1
var1 = 1.9

mu05 = 10.7
var05 = 4.2

mu0 = 9.7
var0 = 53

#pso2 = pso
mus = [muinf, mu10, mu2, mu1, mu05, mu0]
varss = [varinf, var10, var2, var1, var05, var0]

data = pd.DataFrame()

for it in range(len(mus)):
    x = np.random.randn(100000) * np.sqrt(varss[it]) + mus[it]
    data['col_'+str(it)] = x    #plt.plot(x, stats.norm.pdf(x, mus[it], sigma))
pso0 = list(np.load(dirname+'\\pso_arm_0.npy'))

labels = [r'$\alpha \rightarrow +\infty$',r'$\alpha = 10$', r'$\alpha = 2$', r'$\alpha \rightarrow 1^-$',
            r'$\alpha = 0.5$', r'$\alpha \rightarrow 0^+$']

n = 3
f, axes = plt.subplots(2, 3, figsize=(20,15))

for it in range(len(mus)):
    
    axes[it//n, it%n].scatter(samples0, np.multiply(pso0, 1/0.0528), color='black', label='s')       
    sns.distplot(data['col_'+str(it)], hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 2}, 
                      label = labels[it],ax=axes[it//n, it%n], color='orange')
    sns.distplot([1], hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 2}, 
                      label = 'True posterior', color='black',ax=axes[it//n, it%n])    
    axes[it//n, it%n].set_xlim([0,25])
    axes[it//n, it%n].set( ylabel=r'$q(s)$', xlabel=labels[it])

    if it == 0:
        axes[it//n, it%n].legend(['True posterior','Variational posterior'], loc='upper left')
f.savefig('posterior.pdf')    
    
