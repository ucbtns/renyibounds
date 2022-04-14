import numpy as np
import torch, math

import torch.nn as nn
from torch.distributions.normal import Normal
import torch.distributions as D



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


def generate_obs(num_obs, gen_proc_mode1, gen_proc_std1, gen_proc_mode2, gen_proc_std2, gen_proc_weight1):
        obs = []
        for j in range(num_obs):
            y1 = np.random.normal(gen_proc_mode1, gen_proc_std1)
            y2 = np.random.normal(gen_proc_mode2, gen_proc_std2)
            w = np.random.binomial(1, gen_proc_weight1, 1)
            reward = w*y1 + (1-w)*y2
            obs.append(reward.item())
        return obs
    
    
def compute_log_prob(samples, obs, params, mc_samples, prior_mu1, prior_mu2, prior_sigma1, prior_sigma2,
                         mixture_weight_prior, unif_samples=False):

        mu, log_var = params[0], params[1]
        sigma = torch.exp(log_var)
        
        if samples is None:
            if unif_samples:
                samples = torch.arange(0.0, 30.0, 0.01)
            else:
                samples = (torch.randn(mc_samples) * torch.sqrt(sigma) + mu)

        mix = D.Categorical(torch.tensor([mixture_weight_prior, 1-mixture_weight_prior]))
        comp = D.Normal(torch.tensor([prior_mu1, prior_mu2]), torch.tensor([prior_sigma1, prior_sigma2]))
        p0 = D.MixtureSameFamily(mix, comp)
        std_factor = torch.std(torch.tensor([obs], dtype=torch.float32))
        var_factor = std_factor ** 2
        #import pdb; pdb.set_trace()
        factor = Normal(torch.tensor([obs], dtype=torch.float32), var_factor)

        logps = p0.log_prob(samples)
        logfactor = torch.mean(factor.log_prob(samples.reshape(-1, 1)), axis=1)
        logq = -0.5 * torch.log(2 * math.pi * sigma) - 0.5 * (samples - mu) ** 2 / sigma

        return logps, logfactor, logq, samples
    
def get_abcd():
    
    a = np.arange(0.01, 2.0, 0.01)
    b = np.arange(0.0001, 5, 0.002)
    c = np.arange(0.33, 0.45, 0.001)
    d = np.arange(0.0001, 1, 0.002)
    e = np.arange(0.00001, 0.002, 0.00001)
    f = np.arange(0.0001, 20, 0.02)
    g = np.arange(0.001, 5, 0.01)
    h = np.arange(0.001, 1, 0.01)
    
    return a, b, c, d, e, f,g, h 