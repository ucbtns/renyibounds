import numpy as np
import pandas as pd
import torch, wandb

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from common import analytical as al
from common import utils as ut


class univariate_sim:
    
        def __init__(self, dirname, params, y, x, sigma_q, sigma_l, sigma_p):
            
            self.dirname = dirname
            self.params = params

            self.mu_q = params.mu_q 
            self.sigma_q = sigma_q
            self.sigma_l = sigma_l
            self.sigma_p = sigma_p 
            self.y = y
            self.x = x
            self.bl = params.bl
            self.lp = params.lp
            self.ll = params.ll
            self.bp = params.bp
            
            self.plot = params.plot
            
            
        def plot_alpha(self):
            
            rb_value = []
            alphas = []
            for alp in range (0,400):
                alpha = np.exp(alp/100)-1

                if alpha == 1.0:
                   rb_value.append(al.KL(self.mu_q, self.sigma_q, self.sigma_l, self.sigma_p, self.y, self.x))
                else:
                   rb_value.append(al.RB(alpha, self.mu_q, self.sigma_q, self.sigma_l, self.sigma_p, self.y, self.x).item())
        
                alphas.append(alpha)
                
            np.save(self.dirname+'/alphas', alphas)
            np.save(self.dirname+'/bound', rb_value)
            
            if self.plot:
                fig, ax = plt.subplots()
                ax.plot(alphas, rb_value)
                ax.set_yscale('symlog')
                
            return
        
        
        def generate_contour_gamma(self, alphas_p, betas_p):
  
            data = pd.DataFrame()
            i = 0
            for alpha_p in alphas_p:
                bounds = []
                for beta_p in betas_p:
        
                    bound = al.KL_gamma_partial(self.mu_q, self.sigma_q, self.sigma_l, self.sigma_p, 
                                             self.y, self.x, alpha_p, beta_p, self.lp)       
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_gamma' + '.csv')
            return
           
        
        def plot_alpha_p(self):
            
            rb_value = []
            alphas = []
            for alph in range (0,500):
        
                alpha_p = np.exp(alph/100)-0.999
                # Renyi bound:
                rb_value.append(al.KL_gamma_partial(self.mu_q, self.sigma_q, self.sigma_l, self.sigma_p, 
                                                    self.y, self.x, alpha_p, 0.8, self.lp).item())
        
                alphas.append(alpha_p)
        
            np.save(self.dirname+'/alphas_p', alphas)
            np.save(self.dirname+'/rb_value_p', rb_value)
            
            if self.plot:
                fig, ax = plt.subplots()
                ax.plot(alphas, rb_value)
                ax.set_yscale('symlog')
                
            return
        
        
        def generate_contour_gamma_alphas(self, alphas_p,alphas_l):
        
            data = pd.DataFrame()
            i = 0
            for alpha_p in alphas_p:
                bounds = []
                for alpha_l in alphas_l:
        
                    bound = al.KL_gamma(self.mu_q, self.sigma_q, self.sigma_l, self.sigma_p, 
                                        self.y, self.x, alpha_p, self.bp, alpha_l, 
                                        self.bl, self.lp, self.ll)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_gamma_alphas' + '.csv')
            return
        
        
        def generate_contour_alpha_p_muq(self, alphas_p, mu_q):
     
            data = pd.DataFrame()
            i = 0
            for alpha_p in alphas_p:
                bounds = []
                for muq in mu_q:
                    bound = al.KL_gamma_partial(muq, self.sigma_q, self.sigma_l, self.sigma_p, self.y, 
                                                self.x, alpha_p, 0.5, self.lp)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_p_muq' + '.csv')
            return
        
        
        def generate_contour_beta_p_muq(self, betas_p, mu_q):

            data = pd.DataFrame()
            i = 0
            for beta_p in betas_p:
                bounds = []
                for muq in mu_q:
                    bound = al.KL_gamma_partial(muq, self.sigma_q, self.sigma_l, self.sigma_p, 
                                             self.y, self.x, 0.5, beta_p, self.lp)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(betas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_beta_p_muq' + '.csv')
            return
        
        
        def generate_contour_alpha_muq(self, alphas, mu_q):
        
            data = pd.DataFrame()
            i = 0
            for alpha in alphas:
                bounds = []
                for muq in mu_q:
                    bound = al.RB(alpha, muq, self.sigma_q, self.sigma_l, self.sigma_p, self.y, self.x)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_muq' + '.csv')
            return
        
        
        def generate_contour_alpha_p_sigmaq(self, alphas_p, sigma_q):
            
            data = pd.DataFrame()
            i = 0
            for alpha_p in alphas_p:
                bounds = []
                for sigmaq in sigma_q:
                    bound = al.KL_gamma_partial(0.4, sigmaq, self.sigma_l, self.sigma_p, self.y, self.x, alpha_p, 0.5, self.lp)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_p_sigmaq' + '.csv')
            return
        
    
        def generate_contour_beta_p_sigmaq(self, betas_p, sigma_q):
        
            data = pd.DataFrame()
            i = 0
            for beta_p in betas_p:
                bounds = []
                for sigmaq in sigma_q:
                    bound = al.KL_gamma_partial(0.4, sigmaq, self.sigma_l, self.sigma_p, self.y, self.x, 0.5, beta_p, self.lp)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(betas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_beta_p_sigmaq' + '.csv')
            return
        
        
        def generate_contour_alpha_sigmaq(self, alphas, sigma_q):

            data = pd.DataFrame()
            i = 0
            for alpha in alphas:
                bounds = []
                for sigmaq in sigma_q:
                    bound = al.RB(alpha, 0.4, sigmaq, self.sigma_l, self.sigma_p, self.y, self.x)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_sigmaq' + '.csv')
            return
        
        
        def generate_contour_alpha_p_beta_p_sigmaq_const_mean(self, alphas_p, sigma_q):

            data = pd.DataFrame()
            i = 0
            for alpha_p in alphas_p:
                bounds = []
                for sigmaq in sigma_q:
                    bound = al.KL_gamma_partial(0.4, sigmaq, self.sigma_l, self.sigma_p, self.y, 
                                             self.x, alpha_p, alpha_p/5, self.lp)
        
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(alphas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_p_beta_p_sigmaq_const_mean' + '.csv')
            return
        
        
        def generate_contour_alpha_p_beta_p_sigmaq_const_var(self, betas_p, sigma_q):
   
            data = pd.DataFrame()
            i = 0
            for beta_p in betas_p:
                bounds = []
                for sigmaq in sigma_q:
                    bound = al.KL_gamma_partial(0.4, sigmaq, self.sigma_l, self.sigma_p, 
                                                self.y, self.x, np.sqrt(beta_p*5), beta_p, self.lp)    
                    bounds.append(bound.item())
        
                data['bound_'+str(i)] = bounds
                i += 1
                print(i/len(betas_p)*100,"%")
        
            data.to_csv(self.dirname+'/contour_alpha_p_beta_p_sigmaq_const_var' + '.csv')
            return


class multimodal_sim:
    
    def __init__(self, dirname, params, device):
        
        self.dirname = dirname
        self.params = params
        self.device = device
        
        self.arms = params.arms
        self.alphas = [1e-6, 0.5000, 0.99999, 2.0, 10.0, 1e9]
        
        self.policy = []
        self.optimise_policy = []
        
        # Initialise posterior and optim:
        for arm in range(0, self.arms):  
            self.policy.append(ut.VariationalPolicy(params.prior_q, params.sigma_q)); 
            self.optimise_policy.append(optim.Adam(self.policy[arm].parameters(), lr=params.learning_rate))


    def save_distributions(self, arm):

        params = nn.utils.parameters_to_vector(list(self.policy[arm].parameters())).to(self.device, non_blocking=True)
        obs = ut.generate_obs(self.params.num_obs, self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                                self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm], 
                                self.params.mixture_weight_gen_proc[arm])
    
        samples = torch.arange(0.0, 30.0, 0.003)
        logps, logfactor, logq, samples = ut.compute_log_prob(samples, obs, params, 1,
                                                              self.params.prior_mu1[arm], self.params.prior_mu2[arm],
                                                              self.params.prior_sigma1[arm], self.params.prior_sigma2[arm], 
                                                              self.params.mixture_weight_prior[arm], unif_samples=True)
        log_pso = logps + logfactor
        
        ints = torch.mean(torch.exp(log_pso)) * 30
        print("integral of p(s,o) for arm ", arm, ":", ints)
    
        np.save(self.dirname+'/multimodal_samples_'+str(arm),np.array(samples))
        np.save(self.dirname+'/multimodal_rew_'+str(arm),np.array(obs))
        
        ps = torch.exp(logps)
        pos = torch.exp(logfactor)
        pso = torch.exp(log_pso)
        
        np.save(self.dirname+'/multimodal_ps_'+str(arm),np.array(ps))
        np.save(self.dirname+'/multimodal_pos_'+str(arm),np.array(pos))  
        np.save(self.dirname+'/multimodal_pso_'+str(arm),np.array(pso))
        
        return


    def generate_contour(self, arm, means, sigmas):
          
        obs = ut.generate_obs(self.params.num_obs, self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                           self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm],
                           self.params.mixture_weight_gen_proc[arm])
    
        i = 0
        for alpha in self.alphas:
            data = pd.DataFrame()
            samples_unscaled = torch.randn(self.params.mc_samples)
            for mean in means:
                bounds = []
                for sigma in sigmas:
                    params = torch.cat((torch.as_tensor([mean], dtype=torch.float32), torch.log(torch.as_tensor([sigma], dtype=torch.float32)**2)), 0).detach()
    
                    mu_q, log_var = params[0], params[1]
                    sigma_q = torch.exp(log_var)
                    samples = (samples_unscaled * torch.sqrt(sigma_q) + mu_q)
    
                    logps, logfactor, logq, samples = ut.compute_log_prob(samples, obs, params, self.params.mc_samples,
                                                                       self.params.prior_mu1[arm], self.params.prior_mu2[arm],
                                                                       self.params.prior_sigma1[arm],
                                                                       self.params.prior_sigma2[arm],
                                                                       self.params.mixture_weight_prior[arm], 
                                                                       unif_samples=False)
                    
                    bound = ut.compute_policy_loss(self.params.mc_samples, alpha, logps, logfactor, logq)
    
                    bounds.append(bound.detach().item())
    
                data['bound_'+str(i)] = bounds
                i += 1
                print("alpha", alpha, i/len(means)*100,"%")
    
            data.to_csv(self.dirname+'/contour_'+str(alpha) + '.csv')
        return


    def run(self, means, sigmas):
        
        for arm in range(self.arms):
            # Save logp for each arm
            if self.params.save_distributions:
                self.save_distributions(arm)
    
            # Print optimization landscape for arm 1
            if self.params.generate_contour:
                self.generate_contour(arm, means, sigmas)


class mab_sim:
    
    def __init__(self, dirname, params, device):
        
       
        self.dirname = dirname  
        self.params = params
        self.device = torch.device(device)
        self.arms = params.arms


    def pull_arm(self, policy):
        
        samples = []
        with torch.no_grad():
            for arm in range(0, self.arms):
                
                params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(self.device, non_blocking=True)
                mu, log_var = params[0], params[1]
                sigma = torch.exp(log_var)
                samples.append(((torch.randn(1) * torch.sqrt(sigma) + mu) / sigma).item())
                
        arm_pulled = np.argmax(samples)
        
        return arm_pulled
    
    
    def generate_contour(self, arm):
        
        means = np.arange(5.0, 25.0, 0.2)
        sigmas = np.arange(0.001, 5.0, 0.2)
        
        obs = ut.generate_obs(1000, self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                           self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm],
                           self.params.mixture_weight_gen_proc[arm])
        
        data = pd.DataFrame()
        i = 0
        for mean in means:
            bounds = []
            for sigma in sigmas:
                params = torch.cat((torch.as_tensor([mean], dtype=torch.float32), 
                                    torch.as_tensor([sigma], dtype=torch.float32)),0).detach()

                logps, logfactor, logq, samples = ut.compute_log_prob(None, obs, params, self.params.mc_samples,
                                                                   self.params.prior_mu1[arm], self.params.prior_mu2[arm],
                                                                   self.params.prior_sigma1[arm],
                                                                   self.params.prior_sigma2[arm],
                                                                   self.params.mixture_weight_prior[arm],
                                                                   unif_samples=True)

                bound = ut.compute_policy_loss(300, self.params.alpha, logps,
                                               logfactor, logq)

                bounds.append(bound.detach().item())

            data['bound_' + str(i)] = bounds
            i += 1
            print(i / len(means) * 100, "%")

        data.to_csv(self.dirname + '/contour' + '.csv')
        return
    
    
    def evaluate(self, obs, policy, oracle):
        bounds = []
        for arm in range(0, self.arms):
            with torch.no_grad():
                params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(self.device, non_blocking=True)
                logps, logfactor, logq, samples = ut.compute_log_prob(None, obs[arm], params,  self.params.mc_samples,
                                                                               self.params.prior_mu1[arm],
                                                                                self.params.prior_mu2[arm],
                                                                                self.params.prior_sigma1[arm],
                                                                                self.params.prior_sigma2[arm],
                                                                               self.params.mixture_weight_prior[arm])
                bound = ut.compute_policy_loss( self.params.mc_samples,  self.params.alpha, logps,
                                                                 logfactor, logq)
                bounds.append(bound.detach().item())
    
        obs_ev = []
        for _ in range(1000):
            with torch.no_grad():
                arm = self.pull_arm(policy)
                            # print(arm)
                rew = ut.generate_obs(1,  self.params.gen_proc_mode1[arm],  self.params.gen_proc_std1[arm],
                                                self.params.gen_proc_mode2[arm],  self.params.gen_proc_std2[arm],
                                                self.params.mixture_weight_gen_proc[arm])[0]
                obs_ev.append(rew)
                            
                wandb.log({'sharpe_eval': np.mean(obs_ev)/np.var(obs_ev),
                               'regret_eval': np.max(oracle) - np.mean(obs_ev)/np.var(obs_ev)})
    
        return bounds
                
    
    def learn(self):

        policy = []
        optimize_policy = []
    
        # Initialize variational posterior and optimizer
        for arm in range(0, self.arms):
            policy.append(ut.VariationalPolicy(self.params.prior_q, self.params.sigma_q))
            optimize_policy.append(optim.Adam(policy[arm].parameters(), lr=self.params.learning_rate))
    
        post_sharpe = []
        for arm in range(0, self.arms):
            params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(self.device, non_blocking=True)
            obs = ut.generate_obs(self.params.num_obs, self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                               self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm],
                               self.params.mixture_weight_gen_proc[arm])
    
            logps, logfactor, logq, samples = ut.compute_log_prob(None, obs, params, 1,
                                                               self.params.prior_mu1[arm], self.params.prior_mu2[arm],
                                                               self.params.prior_sigma1[arm],
                                                               self.params.prior_sigma2[arm], self.params.mixture_weight_prior[arm],
                                                               unif_samples=True)
    
            log_pso = logps + logfactor
    
            inta = torch.mean(torch.exp(log_pso))
    
            posterior = np.multiply(np.exp(log_pso), 1 / inta)
    
            mean_from_samples = torch.mean(samples * posterior)
            var_from_samples = torch.mean((samples - mean_from_samples) ** 2 * posterior)
    
            post_sharpe.append(mean_from_samples / var_from_samples)

        if self.params.generate_contour:
            self.generate_contour(0)

        oracle = []
        for arm in range(0, self.arms):
            obs = ut.generate_obs(100000, self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                               self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm],
                               self.params.mixture_weight_gen_proc[arm])
            oracle.append(np.mean(obs) / np.var(obs))
        print("sharpe:", oracle)

        obs = [[], [], []]
        rews = []
        arms_pulled = [0,0,0]
        for iter in range(self.params.num_iters):
    
            # Sample each arm once at the beginning
            if iter == 0 or iter == 1:
                arm = 0
            elif iter == 2 or iter == 3:
                arm = 1
            elif iter == 4 or iter == 5:
                arm = 2
            else:
                arm = self.pull_arm(policy)
    
    
            rew = ut.generate_obs(1,self.params.gen_proc_mode1[arm], self.params.gen_proc_std1[arm],
                                    self.params.gen_proc_mode2[arm], self.params.gen_proc_std2[arm],
                                    self.params.mixture_weight_gen_proc[arm])[0]
            obs[arm].append(rew)
    
            # Keep a fixed amount of observations -- memory buffer
            if len(obs[arm]) > self.params.num_obs:
                del obs[arm][0]
    
            rews.append(rew)
            arms_pulled[arm] += 1
    
            # Learn
            print('Learning started')
            if iter > 5:

                for _ in range(self.params.num_updates):
                    
                    params = nn.utils.parameters_to_vector(list(policy[arm].parameters())).to(self.device, non_blocking=True)
    
                    optimize_policy[arm].zero_grad()
    
                    logps, logfactor, logq, samples = ut.compute_log_prob(None, obs[arm], params, self.params.mc_samples,
                                                              self.params.prior_mu1[arm], self.params.prior_mu2[arm], 
                                                              self.params.prior_sigma1[arm],
                                                              self.params.prior_sigma2[arm], 
                                                              self.params.mixture_weight_prior[arm])
    
                    loss_policy = ut.compute_policy_loss(self.params.mc_samples, self.params.alpha, logps, logfactor, logq)
                    loss_policy.backward()
                    optimize_policy[arm].step()
    
                # Log results
                if iter % self.params.log_every == 0:
                    p1 = nn.utils.parameters_to_vector(list(policy[0].parameters())).to(self.device, non_blocking=True)
                    p2 = nn.utils.parameters_to_vector(list(policy[1].parameters())).to(self.device, non_blocking=True)
                    p3 = nn.utils.parameters_to_vector(list(policy[2].parameters())).to(self.device, non_blocking=True)
                    mu1 = p1[0].detach().item()
                    mu2 = p2[0].detach().item()
                    mu3 = p3[0].detach().item()
                    sigma1 = torch.exp(p1[1]).detach().item()
                    sigma2 = torch.exp(p2[1]).detach().item()
                    sigma3 = torch.exp(p3[1]).detach().item()
                    
                    bounds = self.evaluate(obs, policy, oracle)
                    
                    if self.params.print_log:
                        if iter > 2:
        
                            print("iter", iter, "bounds", bounds, "avg regret",
                                  np.max(oracle) - np.mean(rews)/np.var(rews), 'sharpe', np.mean(rews)/np.var(rews),
                                  "mus", [mu1,mu2,mu3], "frac_arm_pulled", arms_pulled,
                                  "Sigmas", [sigma1, sigma2, sigma3])
        
                    wandb.log({'mu1': mu1,
                               'mu2': mu2,
                               'mu3': mu3,
                               'sigma1': sigma1,
                               'sigma2': sigma2,
                               'sigma3': sigma3,
                               'bound1': bounds[0],
                               'bound2': bounds[1],
                               'bound3': bounds[2],
                               'iter': iter,
                               })
        
                if iter % 100 == 0:
        
                    frac_arm1 = arms_pulled[0]/100
                    frac_arm2 = arms_pulled[1]/100
                    frac_arm3 = arms_pulled[2]/100
        
                    wandb.log({'frac_arm1': frac_arm1,
                               'frac_arm2': frac_arm2,
                               'frac_arm3': frac_arm3,
                               'iter2': iter})
        
                    if iter > 2:
                        wandb.log({'avg regret': np.max(oracle) - np.mean(rews) / np.var(rews),
                                   'sharpe': np.mean(rews) / np.var(rews),})
        
                    arms_pulled = [0,0,0]
                    rews = []
        
        
