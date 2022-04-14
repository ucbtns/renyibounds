import pathlib
import sys
import warnings
import os
import torch
from pathlib import Path

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

from common import utils as ut
from common import simulations as sims
from common import flags, config

import yaml
import numpy as np

warnings.filterwarnings('ignore')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def main(dirname, params, device):
    
    
    if params.univariate_sim:
    
        x = np.arange(0,params.x_,params.x_max).reshape(-1,1)
        sigma_l = np.identity(x.shape[0])*params.sl_factor
        y = np.random.multivariate_normal(params.y_factor*x.ravel(),sigma_l )
        sigma_p = np.array(params.sp_factor).reshape(-1,1)
        sigma_q = np.array([[params.sq]])
   
        print("sigma_q must be less than ", 1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x)))
        print("sigma_q:", sigma_q)
        if (sigma_q < 1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x))):
            print("maximum alpha for sq is infinite")
        else:
            print("maximum alpha for sq:", sigma_q/(sigma_q-1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x))))
            
           
        univ = sims.univariate_sim(dirname, params, y, x, sigma_q, sigma_l, sigma_p)
        
        # Varying ranges for beta, alpha, mu, sigma: 
        a, b, c, d, e, f, g, h  = ut.get_abcd()
        
        # Extract data for the relevant plots: 
        univ.generate_contour_gamma(a,a)
        univ.generate_contour_gamma_alphas(a,a)
        univ.generate_contour_alpha_p_muq(b,c)
        univ.generate_contour_beta_p_muq(d,c)
        univ.generate_contour_alpha_muq(f,c)  
        univ.generate_contour_alpha_p_sigmaq(b,e)   
        univ.generate_contour_beta_p_sigmaq(d,e)
        univ.generate_contour_alpha_sigmaq(f,e)    
        univ.generate_contour_alpha_p_beta_p_sigmaq_const_mean(g, e)
        univ.generate_contour_alpha_p_beta_p_sigmaq_const_var(h,e)
        
        
    if params.multivariate_sim:
         # Gives analytical overview for each arm: 
         means = np.arange(params.means_min, params.means_max, params.means_step)
         sigmas = np.arange(params.sigmas_min, params.sigmas_max, params.sigmas_step)
         
         multi = sims.multimodal_sim(dirname, params, device)
          
         multi.run(means, sigmas)
         
    if params.mab_sim:
        
        mab = sims.mab_sim(dirname, params, device)
        mab.learn()

    
if __name__ == '__main__':
    
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    parsed, remaining = flags.Flags(configs=['defaults']).parse(known_only=True)
    
    config = config.Config(configs['defaults'])
   
    for name in parsed.configs: 
        config = config.update(configs[name])
    
    config = flags.Flags(config).parse(remaining)
    
    pa = os.path.dirname(__file__)
    Path(pa + "\\"+ config.id).mkdir(parents=True, exist_ok=True)
    dirname = pa + "\\"+ config.id
    
    import wandb
    os.environ["WANDB_API_KEY"] = config.wandb_key

    if config.wandb_key != '':
      wandb.init(project=config.project,
                 entity=config.userid,
                 name=config.id)
      
      wandb.config.update(config, allow_val_change=True)
      
    device = torch.device("cpu")
    main(dirname, config, device)
    

