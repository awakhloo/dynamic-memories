
import sys
import os 
from tqdm import tqdm
sys.path.append(os.getcwd()) 

import meanfield.hebbian_meanfield as mf
from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, Kvaerno3, Tsit5
import diffrax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import time
import jax
from jax import vmap, jit
jax.config.update("jax_enable_x64", True)

from importlib import reload; reload(mf)


outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'
figdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/figs/sm_figs'
np.random.seed(974324)


# default params
ks = [1.5, 2.0, 2.5, 3.0, 3.5] 
Is = [0.25, 0.5, .75, 1.0, 1.25, 1.5,] 
# ps = [15.]
ps = [10., 25.]
fs = [0.1]
g = 1.3 
# sim params
Ns = [500]
Nsave = 500 # how many to save for lifetime calc
Nsaven = 50 # how many to acutally save 
Nbatch = 10


# time params 
T = 800
dt = 0.5
halt_time = 3/8
halt = T * halt_time
time_params = [(T, halt_time)]
tvec = np.linspace(0,T,int(T/dt))
w=400

# solver 
rtol = 1e-8 
atol = 1e-8

rtol_hi, atol_hi = 1e-9, 1e-9
rtol_lo, atol_lo = 1e-6, 1e-6

# dirs
mod = 'sanity_dt_check'
outpath = outdir + f'/scaling_sims/g_{g}_T_{T}_dt_{dt}_scale_{mod}_sim'
os.makedirs(outpath, exist_ok=True) 

# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}

## start with the params that determine the size of certain arrays and thus cannot be jitted
# for d, N in enumerate(Ns):
#     for (T, halt_time) in time_params:
#         SLV_hi = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol_hi, atol=atol_hi, dt_eval=dt)
#         SLV_lo = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol_lo, atol=atol_lo, dt_eval=dt)
#         solv_hi = jit(vmap(SLV_hi, in_axes=(mapped_inds,)))
#         solv_lo = jit(vmap(SLV_lo, in_axes=(mapped_inds,)))
#         for a, k in enumerate(ks):
#             for c, I in tqdm(enumerate(Is)): 
#                 for b, p in enumerate(ps):
#                     for e, f in enumerate(fs):
#                         omega = 2*np.pi*f
#                         batched_params ={'J' : g/np.sqrt(N) * jnp.array(np.random.randn(Nbatch,N,N)), 
#                                       'Y0' : jnp.stack([jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) for _ in range(Nbatch)]), 
#                                       'thetas' : jnp.array(np.random.uniform(low=0,high=2*np.pi,size=(Nbatch,N,))),
#                                       'g' : g, 'I' : I, 'f' : f, 'k' : k, 
#                                       'p' : p, 'omega' : omega,
#                                       'T' : T, 'halt_time' : halt_time, 
#                                     'halt' : T * halt_time,
#                                      }
#                         Xh = solv_hi(batched_params) # (Batch, time, neuron) 
#                         Xl = solv_lo(batched_params) 
#                         freqs_hi, freqs_lo, lifes_hi, lifes_lo = [], [], [] ,[] 
#                         for i in range(Nbatch): 
#                             fh = mf.get_max_freq(Xh[i], T, halt_time, dt)
#                             lh = mf.get_lifetime(Xh[i], T, dt, halt_time, w=w)
#                             fl = mf.get_max_freq(Xl[i], T, halt_time, dt)
#                             ll = mf.get_lifetime(Xl[i], T, dt, halt_time, w=w)
#                             freqs_hi.append(fh), lifes_hi.append(lh), freqs_lo.append(fl), lifes_lo.append(ll)
#                     results = {'f' : f, 'k' : k, 'p' : p, 'I' : I, 'g' : g,
#                                     'T' : T, 'dt' : dt, 'halt_time' : halt_time, 
#                                'Xh' : Xh[..., :Nsaven], 'Xl' : Xl[..., :Nsaven], 
#                                    'life_hi' : np.array(lifes_hi), 'freqs_hi' : np.array(freqs_hi), 
#                                'life_lo' : np.array(lifes_lo), 'freqs_lo' : np.array(freqs_lo), 
#                                    'N' : N} 
#                     np.save(outpath + f'/k_{round(k,6)}_g_{round(g,6)}_I_{round(I,6)}_p_{round(p,6)}_N_{N}_f_{round(f,6)}_T_{round(T,6)}_halt_{round(halt_time,6)}_comparison.npy', np.array(results))
                            
                            


############
##### EXAMPLE #####

np.random.seed(8375)
g = 1.3 
N = 1000
I = 0.55
f = 0.1 
k = 3.0 
p = 25
phi = jnp.tanh
omega = 2*np.pi*f

T = 600 
halt_time = 1/4
halt = halt_time* T
dt_eval = 0.1
tvec_eval = np.linspace(0,T,int(T/dt_eval))

# random params
J = g/np.sqrt(N) * jnp.array(np.random.randn(N,N))
thetas = jnp.array(np.random.uniform(low=0,high=2*np.pi,size=(N,)))

x0 = np.random.randn(N) 
A0 = np.zeros(N**2)
Y0 = jnp.concatenate([x0,A0])

sim_params = {
              'g' : g, 'I' : I, 'f' : f, 'k' : k, 'p' : p,  'omega' : omega,
              'T' : T, 'halt_time' : halt_time, 
              'J' : J, 'thetas' : thetas, 'halt' : T * halt_time,
             'Y0' : jnp.concatenate([x0,A0])}


lo_res, Al = mf.solve_rand_system(sim_params,T=T, Nsave=500, 
                          rtol=1e-4, atol=1e-4, 
                          dt_eval=dt_eval, N=N,
                         A_times=[T*halt_time])
med_res, Am = mf.solve_rand_system(sim_params,T=T, Nsave=500, 
                          rtol=1e-6, atol=1e-6, 
                          dt_eval=dt_eval, N=N,
                         A_times=[T*halt_time])
hi_res, Ah = mf.solve_rand_system(sim_params,T=T, Nsave=500, 
                          rtol=1e-10, atol=1e-10, 
                          dt_eval=dt_eval, N=N,
                         A_times=[T*halt_time])
teu, eu = mf.solve_euler(sim_params,T=T, Nsave=500, 
                          rtol=1e-10, atol=1e-10, 
                          dt_eval=dt_eval, N=N,
                         dt=0.1)
res = {'eu' : eu, 'hi' : hi_res, 'me' : med_res, 'lo' : lo_res, 'teu' : teu, 'T' : T, 'halt_time': halt_time}
np.save(outdir + '/example_sol.npy', np.array(res))


fig,ax=plt.subplots(figsize=(16,4)) 
colors = ['tomato', 'black', 'grey']
fn=22
lw=2
for i in range(1): 
    ax.plot(tvec_eval-T*halt_time, hi_res[:,i], color=colors[1],alpha=1.,ls='-',lw=lw, label='RK tol=$10^{-10}$' if i ==0 else None)
    ax.plot(tvec_eval-T*halt_time, med_res[:,i], color=colors[0], ls='-', label='RK tol=$10^{-6}$' if i ==0 else None,lw=lw*3, alpha=0.3)
    ax.plot(teu-T*halt_time, eu[:,i], color=colors[2], alpha=1., ls='-', lw=lw, label='Euler dt=$10^{-1}$' if i ==0 else None)
ax.legend(fontsize=1/2*fn, loc='lower left')
ax.set_xlabel("t",fontsize=fn)
fig.savefig(figdir + '/compare_solvers.png', dpi=300, bbox_inches='tight')

