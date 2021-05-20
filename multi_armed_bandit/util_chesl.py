import autograd.numpy as agnp
import numpy as np
import autograd.scipy.stats.norm as agnorm
from autograd import grad
from autograd.misc.optimizers import adam
from scipy.stats import norm as norm
from autograd.scipy.special  import logsumexp
import random, math
from scipy.stats import norm as norm_dist
import scipy.stats as st
import autograd.numpy.random as agnpr
import wandb
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.distributions as D
import math


class VariationalPolicy(nn.Module):

    def __init__(self, prior_mu, prior_sigma):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(prior_mu*np.ones(1, dtype=np.float32)))
        self.log_var = torch.nn.Parameter(torch.log(torch.as_tensor(prior_sigma*np.ones(1, dtype=np.float32))))


def compute_policy_loss(mc_samples, bound, alpha, logps, logfactor, logq):


    if bound == 'Elbo':
        # =============================================================================
        #             lower_bound =  -(agnp.mean(gaussian_entropy(sigma) + (logp0+ logfactor)))
        # =============================================================================
        KL = torch.sum(-0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (mu ** 2 + sigma) / sigma) - \
             torch.sum(-0.5 * torch.log(2 * math.pi * sigma * np.exp(1)))
        lower_bound = -(torch.mean(logfactor) + KL)  # kl(mu, sigma))

    elif bound == 'Renyi':
        # print('ps',logp0 )
        # print('po',logfactor  )
        # print('q',logq  )
        logF = -logq + logps + logfactor
        # print('logF',logF)
        logF = (1 - alpha) * logF
        # print('logF scaled',logF)
        lower_bound = torch.logsumexp(logF, 0) + torch.log(torch.as_tensor(1/mc_samples))
        # print('Lower bound',lower_bound)
        lower_bound = lower_bound / (alpha-1)
        # print('Lower bound scaled',lower_bound)


    elif bound == 'Renyi-half':
        lower_bound = -2 * torch.sum(0.5 * logq + 0.5 * (logps + logfactor))

    elif bound == 'Max':
        logF = logq - (logps + logfactor)
        lower_bound = -torch.max(logF)

    elif bound == 'negMax':
        logF = (logps + logfactor) - logq
        lower_bound = -torch.max(logF)

    return lower_bound
