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
    
    def log_prior(samples_q, v_prior=1.0):
        log_p0 = -0.5 * agnp.log(2 * math.pi * v_prior) - 0.5 * samples_q **2 / v_prior
        return agnp.sum(log_p0)
    
    def bound(params, t):
        
        # Q samples
        mu, sigma = unpack_params(params)

        samples = agnpr.randn(num_samples) * agnp.sqrt(sigma) + mu
        print(samples)
        logp0, logfactor = unnormalised_posterior.unnorm_posterior_log(samples) 
        v_noise = agnp.exp(agnp.log(1.0))
# =============================================================================
#         logp0 = log_prior(samples,1.0)
# =============================================================================
        logq = log_q(samples, mu, v_noise)
        
        if nbound == 'Elbo':     
# =============================================================================
#             lower_bound =  -(agnp.mean(gaussian_entropy(sigma) + (logp0+ logfactor)))   
# =============================================================================
            KL = agnp.sum(-0.5 * agnp.log(2 * math.pi * v_noise) - 0.5 * (mu**2 + sigma) / v_noise) - \
            agnp.sum(-0.5 * agnp.log(2 * math.pi * sigma * np.exp(1)))
            lower_bound = -(agnp.mean(logfactor) + KL) #kl(mu, sigma))
      
        elif nbound == 'Renyi':        
            print('ps',logp0 )
            print('po',logfactor  )
            print('q',logq  )
            logF = logp0 + logfactor - logq 
            print('logF',logF)
            logF = (1-alpha) * logF 
            print('logF scaled',logF)
            lower_bound = -(logsumexp(logF)- logsumexp(logq))
            print('Lower bound',lower_bound)
            lower_bound = lower_bound/(num_samples*(1-alpha)) 
            print('Lower bound scaled',lower_bound)
            
            print('Renyi-negmax', agnp.min(logF))
            print('Renyi', lower_bound)
            print('Renyi-half', -2*agnp.sum(0.5*logq+0.5*(logp0+logfactor)))
            KL = agnp.sum(-0.5 * agnp.log(2 * math.pi * v_noise) - 0.5 * (mu**2 + sigma) / v_noise) - \
            agnp.sum(-0.5 * agnp.log(2 * math.pi * sigma * np.exp(1)))
            elbo = -(agnp.mean(logfactor) + KL) #kl(mu, sigma))
            print('Elbo',elbo)        
            print('Renyi-max', agnp.max(logq - (logp0 + logfactor)))

        elif nbound == 'Renyi-half':            
             lower_bound = -2*agnp.sum(0.5*logq+0.5*(logp0+logfactor))
             
        elif nbound == 'Max':        
            logF =  logq - (logp0 + logfactor) 
            lower_bound = -agnp.max(logF)
            
        elif nbound == 'negMax':        
            logF =  (logp0 + logfactor) -logq  
            lower_bound = -agnp.max(logF)
          
        wandb.log({'Alpha ' + str(k): alpha})
        wandb.log({'Bound ' + str(k): lower_bound._value})
        
        
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
                
    def get_posterior(self, obs, step_size=0.01, num_iters=50,k=0, alpha =1):
        unnorm_posterior = get_unnormalised_posterior(obs, self.prior_mu, self.prior_sigma, self.likelihood_sigma)
        variational_objective, gradient, unpack_params = variationalbound(unnorm_posterior, self.gradient_samples, self.bound, self.alpha,k)
        init_var_params = agnp.array([self.prior_mu, self.prior_sigma])

        return init_var_params, gradient
    
    