import numpy as np
import torch
import torch.nn as nn


class VariationalPolicy(nn.Module):

    def __init__(self, prior_mu, prior_sigma):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.as_tensor(prior_mu*np.ones(1), dtype=torch.float32))
        self.log_var = torch.nn.Parameter(torch.log(torch.as_tensor(prior_sigma*np.ones(1), dtype=torch.float32)))


def compute_policy_loss(mc_samples, alpha, logps, logfactor, logq):

    logF = -logq + logps + logfactor
    logF = (1 - alpha) * logF
    lower_bound = torch.logsumexp(logF, 0) + torch.log(torch.as_tensor(1/mc_samples))
    lower_bound = lower_bound / (alpha-1)

    return lower_bound
