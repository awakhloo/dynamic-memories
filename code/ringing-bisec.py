

'''
fig 1 entrainment 
'''
import sys
import os 
from tqdm import tqdm
import gc
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

outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'
np.random.seed(1957)


from importlib import reload; reload(mf)
reload(mf)


f=0.1
p = 25
g = 1.3 
omega = f * 2 * np.pi
ks = np.linspace(0,3.5,36)


T = 5_000 # measure time in interval in units of frequency. this is done to avoid edge effects when calculating the autocov. function 
# T = 2500
halt_time = 1.5
init_time = 0.
dt=0.1
Nbatch = 9

# sim params
N = 500
Nsave=50

# bisec params
reps_bisec = 8
I_lower = 0.25
I_upper = 2.0
burn = int(T/dt/2) 

# tolerance for crit amp 
tol, tol_m = 1e-3, 1e-1

# ode solver tols 
rtol, atol = 1e-6, 1e-6
max_step = 50_000

# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}
SLV = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol, atol=atol, dt_eval=dt, max_steps=max_step)
solv = jit(vmap(SLV, in_axes=(mapped_inds,)))

params ={
          'g' : g,  'f' : f,
          'p' : p, 'omega' : omega,
          'T' : T, 'halt_time' : halt_time, 
        'halt' : T * halt_time,
                     }

for k in tqdm(ks): 
    params['k'] = k 
    I_crit, bounds, Is, X = mf.get_crit_amp_sim(solv, params, N, T, dt, Nbatch, reps_bisec, 
                                                I_lower, I_upper, tol, tol_m,init_time, burn, phi=np.tanh) 
    outpath = outdir + f'/entrainment_bisection/T_{T}_entrain_p_{p}_f_{f}_g_{g}_N_{N}_Nbatch_{Nbatch}/'
    os.makedirs(outpath, exist_ok=True)
    np.save(outpath + f'/crit_{k}.npy', 
            np.array([I_crit, k, g, p, f]))


