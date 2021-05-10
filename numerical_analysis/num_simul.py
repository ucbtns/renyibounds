# =============================================================================
# @author: Noor Sajid
# @purpose: Section 4.0 for the Renyi Bound paper
# =============================================================================

import numpy as np
import math
import matplotlib.pyplot as plt
np.random.seed(1)

## ELBO - gaussian-gamma system, with a gaussian approximate density:
# ===================================
nlg = lambda x: np.log(x + 1e-32) 
k_ELBO = lambda lq, sq,lp,sp,ll,sl,x: sp*lp + (x.T.dot(ll*sl).dot(x)) - lq*sq
mu_ELBO = lambda k, muq, sq,lq, ll, sl, y, x: k**-1*( x.T.dot(ll*sl).dot(y)-lq*sq*muq)

def ELBO(muq, lq, sq, aq, bq,
         ll, sl, al, bl, 
         lp, sp, ap, bp, y, x):
    
    k = k_ELBO(lq, sq,lp,sp,ll,sl,x)
    mu = mu_ELBO(k,muq, sq,lq, ll, sl, y, x)
    
    M = sl.shape[0]      
    fr = 0.5*nlg((np.linalg.det((ll*sl))*np.linalg.det(lp*sp))/np.linalg.det((lq*sq))) -nlg(math.gamma(al)*math.gamma(ap))
    sr =  al*nlg(bl*ll) + ap*nlg(bp*lp) - nlg(lp*ll)-M*nlg(2*np.pi) - lp*bp - ll*bl - 0.5*ll*y.T.dot(sl).dot(y)
    tr = -0.5*((muq-mu).T.dot(k).dot(muq-mu) + np.trace(k.dot(lq*sq)))
    er = 0.5*((mu).T.dot(k).dot(mu) + lq*muq.T.dot(sq).dot(muq))
    
    return fr + sr +tr + er

## Renyi bound - gaussian system:
# ===================================

sigma_RB = lambda alpha, lam, sq: ((1-alpha)*lam + alpha/sq)**-1
lam_RB = lambda sp, sl, x: 1/sp +  x.T.dot(np.linalg.inv(sl)).dot(x)
mu_RB = lambda alpha, s, muq, sq, sl, y, x: s*(alpha*(np.linalg.inv(sq))*muq + (1-alpha)*x.T.dot(np.linalg.inv(sl)).dot(y))


def RB(alpha, muq, sq, sl, sp, y, x ):
    
    lam = lam_RB(sp, sl, x)
    s = sigma_RB(alpha, lam, sq)
    mu = mu_RB(alpha, s, muq, sq, sl, y, x )
    
    tr1 = -0.5*(alpha*(muq**2/sq) + (1-alpha)*y.T.dot(np.linalg.inv(sl)).dot(y) - mu**2/s)
    
    
    tr2 = np.linalg.det(s)/((np.linalg.det(sq)**alpha) *(np.linalg.det(sp)**(1-alpha))
                                    *(np.linalg.det(sl)**(1-alpha)) * (2*np.pi)**((1-alpha)*y.shape[0]))
    
# =============================================================================
#     tr1 = -0.5*(alpha*muq.T.dot(np.linalg.inv(sq)).dot(muq) + (1-alpha)*y.T.dot(np.linalg.inv(sl)).dot(y) - mu.T.dot(np.linalg.inv(s)).dot(mu))            
#     tr2 = np.linalg.det(s)/ ((np.linalg.det(sq)**alpha) *(np.linalg.det(sp)**(1-alpha))
#                                     *(np.linalg.det(sl)**(1-alpha)) * (2*np.pi)**((1-alpha)*y.shape[0]))
#     
# =============================================================================
    return (0.5*nlg(tr2) + tr1)/(1-alpha)
 
def gelbo(muq,sq,sl,sp,y,x):
    #lam = lam_(sp, sl, x)  
    #  (np.log(sq/sp)) + (sq**2+(muq-mup)**2)/2*sp**2 + 1/2

    import math
    KL =np.sum(-0.5 * nlg(2 * math.pi * sp) - 0.5 * (muq**2 + sq) / sp) - \
                np.sum(-0.5 * nlg(2 * math.pi * sq * np.exp(1)))
    return (np.mean(sl) + KL)


# Simulation 1:
# ===================================================
x = np.arange(0,20,1.1).reshape(-1,1)
sl = np.identity(x.shape[0])*5
y = np.random.multivariate_normal(0.4*x.ravel(), sl)

sp = np.array(5).reshape(-1,1)
mq = np.array(0.0).reshape(-1,1)
sq = np.array(1e-4).reshape(-1,1)

bq = 0.8
bp = 0.8
bl= 0.8

means = []
prior = []
rb_value = []
elbo_value1 = []
elbo_value2 = []
elbo_value3 = []
pr = 0.8
al = pr
ap = pr
aq = pr
lq = pr
lp = pr
ll = pr 

prr = 0.8

m = np.array(10)
#np.arange(0.1,240,0.5)
alpha = [0.1, 0.2, 0.3, 0.4,0.6, 0.7,0.8,0.9, 1.0]
prior =[0, 0.2,0.6, 1.4, 1.6, 2.8, 3.4, 3.8,4.0]
for (a, pr) in zip(alpha, prior):  
        means.append(m)  
        lq = pr
        lp = pr 
        ll = pr 
        bq = prr
        bp = prr
        bl = prr

        # Renyi bound:
        if a == 1.0:
           rb_value.append(np.array(gelbo(m.reshape(-1,1),sq,sl,sp,y,x)).reshape(-1,1))
        else:
           rb_value.append(RB(a,m.reshape(-1,1), sq, sl, sp, y, x))
        
        # ELBO: 
        elbo_value2.append(ELBO(m.reshape(-1,1), lq, sq, aq, bq,ll, sl, al, bl, lp, sp, ap, bp, y, x))  
        bq = pr
        bp = pr
        bl = pr
        elbo_value1.append(ELBO(m.reshape(-1,1), lq, sq, aq, bq,ll, sl, al, bl, lp, sp, ap, bp, y, x))
        lq = prr
        lp = prr
        ll = prr        
        
        elbo_value3.append(ELBO(m.reshape(-1,1), lq, sq, aq, bq,ll, sl, al, bl, lp, sp, ap, bp, y, x))

rb_value = np.concatenate(rb_value)
elbo_value1 = np.concatenate(elbo_value1)
elbo_value2 = np.concatenate(elbo_value2)
elbo_value3 = np.concatenate(elbo_value3)


# Plot:
# =====================================================
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4), dpi=1000)
axes[0].plot(alpha,rb_value/2, linewidth=2.5, color ='black')
axes[0].set_xlim([0,1.05])
axes[0].set_ylim([-102000,0.1])
axes[0].set_ylabel('Renyi Bound (nats)')
axes[0].set_xlabel(r'$\alpha$ value')
axes[1].set_ylabel('Negative Free Energy (nats)')
axes[1].set_xlabel(r"$\lambda_K$ value")
axes[1].plot(prior,elbo_value2, linewidth=2.5, color ='black')
axes[1].set_xlim([0,4.1])
axes[1].set_ylim([-102000,0.1])
axes[2].set_ylabel('Negative Free Energy (nats)')
axes[2].set_xlabel(r"$\beta_K$ value")
axes[2].plot(prior,elbo_value3, linewidth=2.5, color ='black')
axes[2].set_xlim([0,4.1])
axes[2].set_ylim([-102000,0.1])

axes[3].set_ylabel('Negative Free Energy (nats)')
axes[3].set_xlabel(r"$\beta_K & \lambda_K$ value")
axes[3].plot(prior,elbo_value1, linewidth=2.5, color ='black')
axes[3].set_xlim([0,4.1])
axes[3].set_ylim([-102000,0.1])
fig.tight_layout()


# Simulation 2:
# ========================================================

x = np.arange(0,20,1.1).reshape(-1,1)
sl = np.identity(x.shape[0])*0.2
y = np.random.multivariate_normal(0.4*x.ravel(), sl)

sp = np.array(1.0).reshape(-1,1)
mq = np.array(0.0).reshape(-1,1)
sq = np.array(0.002).reshape(-1,1)

bq = 0.8
bp = 0.8
bl= 0.8

rb_valueS = np.zeros((20,9))
elbo_valueS = np.zeros((20,9))

elbo_value2 = np.zeros((20,9))
pr = 0.8
al = pr
ap = pr
aq = pr
lq = pr
lp = pr
ll = pr 

m = np.array(10)
#np.arange(0.1,240,0.5)
alpha = [0.1, 0.2, 0.3, 0.4,0.6, 0.7,0.8,0.9, 1.0]
prior =[0.2, 0.4,0.6, 0.8, 1.0, 1.2, 1.4, 1.6,2.8]
means = np.arange(-100, 100, 10)

for i, (a, pr) in enumerate(zip(alpha, prior)):  
    for n, m in enumerate(means):
        lq = pr
        lp = pr 
        ll = pr 
        bq = 0.8
        bp = 0.8
        bl = 0.8

        # Renyi bound:
        if a == 1.0:
           rb_valueS[n,i] = gelbo(m.reshape(-1,1),sq,sl,sp,y,x)
        else:
           rb_valueS[n,i] = RB(a,m.reshape(-1,1), sq, sl, sp, y, x)
        
        # ELBO: 
        elbo_valueS[n,i] = ELBO(m.reshape(-1,1), lq, sq, aq, bq,ll, sl, al, bl, lp, sp, ap, bp, y, x)
         # ELBO: 
        bq = pr
        bp = pr
        bl = pr
        lq = 0.8
        lp = 0.8
        ll = 0.8  
        elbo_value2[n,i] = ELBO(m.reshape(-1,1), lq, sq, aq, bq,ll, sl, al, bl, lp, sp, ap, bp, y, x)
        

im = plt.imshow(rb_valueS/2, cmap=plt.cm.RdBu, extent=(0,1,0,1), interpolation='bilinear')
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 4), dpi=1000)
axes[0].imshow(rb_valueS, cmap=plt.cm.RdBu, extent=(-100,100,-100,100), interpolation='bilinear')
axes[0].set_ylabel(r'$u_q$')
axes[0].set_xlabel(r'$\alpha$ value')
axes[1].imshow(elbo_valueS, cmap=plt.cm.RdBu, extent=(-100,100,-100,100), interpolation='bilinear')
axes[1].set_xlabel(r"$\lambda_K$ value")
axes[2].imshow(elbo_value2, cmap=plt.cm.RdBu, extent=(-100,100,-100,100), interpolation='bilinear')
axes[2].set_xlabel(r"$\beta_K$ value")
fig.tight_layout()
plt.colorbar(im)
plt.show()








