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


class VariationalPolicy(nn.Module):

    def __init__(self, prior_mu, prior_sigma):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(prior_mu*np.ones(1, dtype=np.float32)))
        self.log_std = torch.nn.Parameter(torch.log(torch.as_tensor(prior_sigma*np.ones(1, dtype=np.float32))))
