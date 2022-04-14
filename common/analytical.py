# =============================================================================
# Bounds
# =============================================================================

import numpy as np
import math


nlg = lambda x: np.log(x)
sigma_RB = lambda alpha, sp, sl, sq, x: ((1-alpha)*(1/sp + x.T.dot(np.linalg.inv(sl)).dot(x)) + alpha/sq)**-1
mu_RB = lambda alpha, s, muq, sq, sl, y, x: s*(alpha*(1/sq)*muq + (1-alpha)*x.T.dot(np.linalg.inv(sl)).dot(y))
sigma_KL = lambda sp, sl, sq, x: (1/sp + x.T.dot(np.linalg.inv(sl)).dot(x) - 1/sq)**-1
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

    tr1 = 0.5*nlg(sq / (sp * np.linalg.det(sl))) - 0.5 * y.shape[0] * nlg(2 * np.pi)

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
