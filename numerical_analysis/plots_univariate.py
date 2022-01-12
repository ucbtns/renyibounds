# =============================================================================
# @author: Noor Sajid
# @purpose: Section 4.0 for the Renyi Bound paper
# =============================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
np.random.seed(1)

nlg = lambda x: np.log(x)
sigma_RB = lambda alpha, sp, sl, sq, x: ((1-alpha)*(1/sp +  x.T.dot(np.linalg.inv(sl)).dot(x)) + alpha/sq)**-1
mu_RB = lambda alpha, s, muq, sq, sl, y, x: s*(alpha*(1/sq)*muq + (1-alpha)*x.T.dot(np.linalg.inv(sl)).dot(y))
sigma_KL = lambda sp, sl, sq, x: (1/sp +  x.T.dot(np.linalg.inv(sl)).dot(x) - 1/sq)**-1
mu_KL = lambda s_kl, muq, sq, sl, y, x: s_kl*(-(1/sq)*muq + x.T.dot(np.linalg.inv(sl)).dot(y))


def RB(alpha, muq, sq, sl, sp, y, x):

    #compute sigma alpha
    s = sigma_RB(alpha, sp, sl, sq, x)
    mu = mu_RB(alpha, s, muq, sq, sl, y, x )

    tr1 = 0.5 * nlg(sq/(sp* np.linalg.det(sl))) - 0.5* y.shape[0]*nlg(((2*np.pi))) + 0.5 / (alpha - 1) * nlg(sq/s)

    tr2 = 0.5/(alpha-1)*(alpha*(muq**2/sq) + (1-alpha)*y.T.dot(np.linalg.inv(sl)).dot(y) - mu**2/s)

    return tr1 + tr2


def KL(muq, sq, sl, sp, y, x):

    s_kl = sigma_KL(sp, sl, sq, x)
    mu_kl = mu_KL(s_kl, muq, sq, sl, y, x)

    tr1 = 0.5*nlg(sq / (sp * np.linalg.det(sl))) - 0.5 * y.shape[0] * nlg(((2 * np.pi)))

    tr2 = - 0.5*(y.T.dot(np.linalg.inv(sl)).dot(y) - muq ** 2 / sq - mu_kl **2 / s_kl + (muq - mu_kl) ** 2 / s_kl)

    tr3 = - 0.5* sq / s_kl

    kl = tr1 + tr2 + tr3
    return kl


def KL_gamma(muq, sq, sl, sp, y, x, ap, bp, al, bl, lp, ll):

    s_kl = sigma_KL(sp, sl, sq, x)
    mu_kl = mu_KL(s_kl, muq, sq, sl, y, x)

    tr1 = 0.5*nlg(sq / (sp * np.linalg.det(sl))) - 0.5 * y.shape[0] * nlg(((2 * np.pi)))

    tr2 = - 0.5*(y.T.dot(np.linalg.inv(sl)).dot(y) - muq ** 2 / sq - mu_kl **2 / s_kl + (muq - mu_kl) ** 2 / s_kl)

    tr3 = - 0.5* sq / s_kl

    tr4 = (al-1)*nlg(ll) + (ap-1)*nlg(lp) + (al)*nlg(bl) + (ap)*nlg(bp) - nlg(math.gamma(al)) - nlg(math.gamma(ap))

    kl = tr1 + tr2 + tr3 + tr4
    return kl


def KL_gamma_partial(muq, sq, sl, sp, y, x, ap, bp, lp):

    s_kl = sigma_KL(sp, sl, sq, x)
    mu_kl = mu_KL(s_kl, muq, sq, sl, y, x)

    tr1 = 0.5*nlg(sq / (sp * np.linalg.det(sl))) - 0.5 * y.shape[0] * nlg(((2 * np.pi)))

    tr2 = - 0.5*(y.T.dot(np.linalg.inv(sl)).dot(y) - muq ** 2 / sq - mu_kl **2 / s_kl + (muq - mu_kl) ** 2 / s_kl)

    tr3 = - 0.5* sq / s_kl

    tr4 = (ap-1)*nlg(lp) + (ap)*nlg(bp) - nlg(math.gamma(ap))

    kl = tr1 + tr2 + tr3 + tr4
    return kl


# Simulation 1:
# ===================================================


x = np.arange(0,20,1.1).reshape(-1,1)
sigma_l = np.identity(x.shape[0])*1
y = np.random.multivariate_normal(0.4*x.ravel(), sigma_l)
sigma_p = np.array(1).reshape(-1,1)
sigma_q = np.array([[0.0001]])
mu_q = 1

bp = 0.8
bl = 0.8
al = 0.8
ap = 0.8
lp = 0.8
ll = 0.01


print("sigma_q must be less than ", 1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x)))
print("sigma_q:", sigma_q)
if (sigma_q < 1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x))):
    print("maximum alpha for sq is infinite")
else:
    print("maximum alpha for sq:", sigma_q/(sigma_q-1/(1/sigma_p + x.T.dot(np.linalg.inv(sigma_l)).dot(x))))

import os
dirname = os.path.dirname(__file__)


def plot_alpha():
    rb_value = []
    alphas = []
    for alp in range (0,400):

        alpha = np.exp(alp/100)-1
        # Renyi bound:
        if alpha == 1.0:
           rb_value.append(KL(mu_q, sigma_q, sigma_l, sigma_p, y, x))
        else:
           rb_value.append(RB(alpha, mu_q, sigma_q, sigma_l, sigma_p, y, x).item())

        alphas.append(alpha)
    print("bound", rb_value)
    print("alpha", alphas)
    np.save(dirname+'/alphas', alphas)
    np.save(dirname+'/bound', rb_value)

    fig, ax = plt.subplots()
    ax.plot(alphas, rb_value)
    ax.set_yscale('symlog')
    return


def generate_contour_gamma():
    alphas_p = np.arange(0.01, 2.0, 0.01)
    betas_p = np.arange(0.01, 2.0, 0.01)


    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha_p in alphas_p:
        bounds = []
        for beta_p in betas_p:

            bound = KL_gamma_partial(mu_q, sigma_q, sigma_l, sigma_p, y, x, alpha_p, beta_p, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas_p)*100,"%")

    data.to_csv(dirname+'/contour_gamma' + '.csv')
    return


def plot_alpha_p():
    rb_value = []
    alphas = []
    for alph in range (0,500):

        alpha_p = np.exp(alph/100)-0.999
        # Renyi bound:
        rb_value.append(KL_gamma_partial(mu_q, sigma_q, sigma_l, sigma_p, y, x, alpha_p, 0.8, lp).item())

        alphas.append(alpha_p)
    print("bound", rb_value)
    print("alpha", alphas)
    np.save(dirname+'/alphas_p', alphas)
    np.save(dirname+'/rb_value_p', rb_value)

    fig, ax = plt.subplots()
    ax.plot(alphas, rb_value)
    ax.set_yscale('symlog')
    return


def generate_contour_gamma_alphas():
    alphas_p = np.arange(0.01, 2.0, 0.01)
    alphas_l = np.arange(0.01, 2.0, 0.01)

    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha_p in alphas_p:
        bounds = []
        for alpha_l in alphas_l:

            bound = KL_gamma(mu_q, sigma_q, sigma_l, sigma_p, y, x, alpha_p, bp, alpha_l, bl, lp, ll)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas_p)*100,"%")

    data.to_csv(dirname+'/contour_gamma_alphas' + '.csv')
    return


def generate_contour_alpha_p_muq():
    alphas_p = np.arange(0.0001, 5, 0.002)

    mu_q = np.arange(0.33, 0.45, 0.001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha_p in alphas_p:
        bounds = []
        for muq in mu_q:
            bound = KL_gamma_partial(muq, sigma_q, sigma_l, sigma_p, y, x, alpha_p, 0.5, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas_p)*100,"%")

    data.to_csv(dirname+'/contour_alpha_p_muq' + '.csv')
    return


def generate_contour_beta_p_muq():
    betas_p = np.arange(0.0001, 1, 0.002)
    mu_q = np.arange(0.33, 0.45, 0.001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for beta_p in betas_p:
        bounds = []
        for muq in mu_q:
            bound = KL_gamma_partial(muq, sigma_q, sigma_l, sigma_p, y, x, 0.5, beta_p, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(betas_p)*100,"%")

    data.to_csv(dirname+'/contour_beta_p_muq' + '.csv')
    return


def generate_contour_alpha_muq():

    alphas = np.arange(0.0001, 20, 0.02)
    mu_q = np.arange(0.33, 0.45, 0.001)


    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha in alphas:
        bounds = []
        for muq in mu_q:
            bound = RB(alpha, muq, sigma_q, sigma_l, sigma_p, y, x)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas)*100,"%")

    data.to_csv(dirname+'/contour_alpha_muq' + '.csv')
    return


def generate_contour_alpha_p_sigmaq():
    alphas_p = np.arange(0.0001, 5, 0.002)
    sigma_q = np.arange(0.00001, 0.002, 0.00001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha_p in alphas_p:
        bounds = []
        for sigmaq in sigma_q:
            bound = KL_gamma_partial(0.4, sigmaq, sigma_l, sigma_p, y, x, alpha_p, 0.5, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas_p)*100,"%")

    data.to_csv(dirname+'/contour_alpha_p_sigmaq' + '.csv')
    return



def generate_contour_beta_p_sigmaq():
    betas_p = np.arange(0.0001, 1, 0.002)
    sigma_q = np.arange(0.00001, 0.002, 0.00001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for beta_p in betas_p:
        bounds = []
        for sigmaq in sigma_q:
            bound = KL_gamma_partial(0.4, sigmaq, sigma_l, sigma_p, y, x, 0.5, beta_p, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(betas_p)*100,"%")

    data.to_csv(dirname+'/contour_beta_p_sigmaq' + '.csv')
    return


def generate_contour_alpha_sigmaq():

    alphas = np.arange(0.0001, 20, 0.02)
    sigma_q = np.arange(0.00001, 0.002, 0.00001)


    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha in alphas:
        bounds = []
        for sigmaq in sigma_q:
            bound = RB(alpha, 0.4, sigmaq, sigma_l, sigma_p, y, x)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas)*100,"%")

    data.to_csv(dirname+'/contour_alpha_sigmaq' + '.csv')
    return


def generate_contour_alpha_p_beta_p_sigmaq_const_mean():
    alphas_p = np.arange(0.001, 5, 0.01)
    sigma_q = np.arange(0.00001, 0.002, 0.00001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for alpha_p in alphas_p:
        bounds = []
        for sigmaq in sigma_q:
            bound = KL_gamma_partial(0.4, sigmaq, sigma_l, sigma_p, y, x, alpha_p, alpha_p/5, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(alphas_p)*100,"%")

    data.to_csv(dirname+'/contour_alpha_p_beta_p_sigmaq_const_mean' + '.csv')
    return



def generate_contour_alpha_p_beta_p_sigmaq_const_var():
    betas_p = np.arange(0.001, 1, 0.01)
    sigma_q = np.arange(0.00001, 0.002, 0.00001)
    import pandas as pd

    data = pd.DataFrame()
    i = 0
    for beta_p in betas_p:
        bounds = []
        for sigmaq in sigma_q:
            bound = KL_gamma_partial(0.4, sigmaq, sigma_l, sigma_p, y, x, np.sqrt(beta_p*5), beta_p, lp)

            bounds.append(bound.item())

        data['bound_'+str(i)] = bounds
        i += 1
        print(i/len(betas_p)*100,"%")

    data.to_csv(dirname+'/contour_alpha_p_beta_p_sigmaq_const_var' + '.csv')
    return



#plot_alpha()
#plot_alpha_p()
#generate_contour_gamma()
generate_contour_alpha_p_muq()
#generate_contour_gamma_alphas()
generate_contour_beta_p_muq()
generate_contour_alpha_muq()
generate_contour_alpha_p_sigmaq()
generate_contour_beta_p_sigmaq()
generate_contour_alpha_sigmaq()
#generate_contour_alpha_p_beta_p_sigmaq_const_mean()
#generate_contour_alpha_p_beta_p_sigmaq_const_var()