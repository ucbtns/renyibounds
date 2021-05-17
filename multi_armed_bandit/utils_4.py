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
m= st.multivariate_normal([2,5], [0.75,0.25])
import wandb


class MoGMAB:   
    def __init__(self, mu1, sigma1, mu2, sigma2):
        
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
       
    def draw(self, k):
        y1 = np.random.normal(self.mu1[k], self.sigma1[k])
        y2 = np.random.normal(self.mu2[k], self.sigma2[k])
        w = np.random.binomial(1, .6,1)        
        reward = w*y1 + (1-w)*y2                  
        regret = w*(np.max(self.mu1) - self.mu1[k]) +  (1-w)*(np.max(self.mu2)- self.mu2[k])
        return reward, regret

def sample_proposal(X, Sigma=1):   
    sample = np.random.normal(X,Sigma) # x' ~ q(x'|x)        
    return sample

def proposal(X, mu,sigma):   
    mn = norm(mu,sigma);
    p = mn.logpdf(X);
    return p

def target_sample(mu,sigma,weights):
  return random.choices(population=[norm(mu[0],sigma[0]).rvs(1),norm(mu[1],sigma[1]).rvs(1)],  weights=weights, k=1)

def target(X,mu,sigma,weights):
  K = len(weights)
  n = np.size(X,0)
  p = np.zeros([n,1])

  for k in range(K):
      mn = norm(mu[k],sigma[k]);
      p = p + weights[k]*mn.pdf(X)
  return p


class get_unnormalised_posterior:
    
    def __init__(self, obs, prior_mu, prior_std, likelihood_std):
        self.obs = obs
        self.prior_mu = prior_mu
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        
    def unnorm_posterior_log(self, samples):                
        prior_density = agnorm.logpdf(samples, loc=self.prior_mu, scale=self.prior_std)

        likelihood_density = agnp.sum(agnorm.logpdf(samples.reshape(-1,1), loc=self.obs, scale=self.likelihood_std), axis=1)
        return prior_density, likelihood_density

def variationalbound(unnormalised_posterior, num_samples, nbound, alpha,k):

    def unpack_params(params):
        mu, log_sigma = params[0], params[1]
        return mu, agnp.exp(log_sigma)

    def gaussian_entropy(sigma):
        return agnp.log(sigma*agnp.sqrt(2*agnp.pi*agnp.e))
    
    def log_q(samples, mu, sigma):
        log_q = -0.5 * agnp.log(2 * math.pi * sigma) - 0.5 * (samples - mu) **2 / sigma
        return log_q
    
    def bound(params, t):
        
        mu, sigma = unpack_params(params)
        samples = agnpr.randn(num_samples) * sigma + mu
        logp0, logfactor = unnormalised_posterior.unnorm_posterior_log(samples) 
        logq = log_q(samples, mu, sigma)
        
        if nbound == 'Elbo':     
            lower_bound =  -(agnp.mean(gaussian_entropy(sigma) + (logp0+ logfactor)))   
      
        elif nbound == 'Renyi':            
            logF = logp0 + logfactor - logq 
            logF = (1-alpha) * logF 
            lower_bound = -(logsumexp(logF)- logsumexp(logq))
            lower_bound = lower_bound/(1-alpha) 

        elif nbound == 'Renyi-half':            
             lower_bound = -2*agnp.sum(0.5*logq+0.5*(logp0+logfactor))
             
        elif nbound == 'Max':        
            logF =  logp0 + logfactor -logq  
            lower_bound = -agnp.max(logF)
            
        wandb.log({'Bound ' + str(k): lower_bound._value})
        
        print(lower_bound)
        return lower_bound
   
    gradient = grad(bound)
    
    return bound, gradient, unpack_params


class variationalinference:
    
    def __init__(self, prior_mu, prior_sigma, likelihood_sigma, bound, alpha, gradient_samples):

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.post_mu = prior_mu
        self.post_sigma = prior_sigma
        self.likelihood_sigma = likelihood_sigma
        self.bound = bound
        self.gradient_samples = gradient_samples        
        self.alpha = alpha
                
    def get_posterior(self, obs, step_size=0.01, num_iters=50,k=0):
        unnorm_posterior = get_unnormalised_posterior(obs, self.prior_mu, self.prior_sigma, self.likelihood_sigma)
        variational_objective, gradient, unpack_params = variationalbound(unnorm_posterior, self.gradient_samples, self.bound, self.alpha,k)
        init_var_params = agnp.array([self.prior_mu, self.prior_sigma])
        
        variational_params = adam(gradient, init_var_params, step_size=step_size, num_iters=num_iters)
        self.post_mu, self.post_sigma = variational_params[0], np.exp(variational_params[1])

        return variational_params, norm_dist(self.post_mu, self.post_sigma)
    
    
    