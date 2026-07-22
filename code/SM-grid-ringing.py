
import sys
import os 
from tqdm import tqdm
import gc
import pandas as pd 
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
np.random.seed(6432)



# p = 15. 
p = 25 
g = 1.8
f = 0.05
omega = 2 * np.pi * f
# ks = np.linspace(0, 3.5, 61)
# Is = np.linspace(0, 2.0, 61)
ks = np.linspace(0, 3.5, 61)
Is = np.linspace(0, 2.0, 61)


# time params 
T =  600  # 500 #
dt = 0.1
halt_time =  1/6 #1/4 #0.5


# # sim params

# N = 1000
N=500
Nbatch = 50
Nsave = 20


# lifetime params
w=400
skip = 50 # +/- 2.5s lifetime measure


rtol = 1e-6
atol = 1e-6

dfs = []



outpath = outdir + f'/grid/grid_N_{N}_Nb_{Nbatch}_dt_{dt}_w_{w}_T_{T}_omega_{omega}_SM'
print(outpath,flush=True) 
os.makedirs(outpath,exist_ok=True)


# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}
SLV = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol, atol=atol, dt_eval=dt)
solv = jit(vmap(SLV, in_axes=(mapped_inds,)))

for i, k in enumerate(ks):
    for j, I in tqdm(enumerate(Is)):
        omega = 2*np.pi*f
        t0 = time.time()
        batched_params ={'J' : g/np.sqrt(N) * jnp.array(np.random.randn(Nbatch,N,N)), 
                      'Y0' : jnp.stack([jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) for _ in range(Nbatch)]), 
                      'thetas' : jnp.array(np.random.uniform(low=0,high=2*np.pi,size=(Nbatch,N,))),
                      'g' : g, 'I' : I, 'f' : f, 'k' : k, 
                      'p' : p, 'omega' : omega,
                      'T' : T, 'halt_time' : halt_time, 
                    'halt' : T * halt_time,
                     }
        Xh, *_ = solv(batched_params) # (Batch, time, neuron) 
        freqs, lifes = [], [] 
        t1 = time.time()
        for b in range(Nbatch):
            X = Xh[b]
            freq = mf.get_max_freq(X, T, halt_time, dt)
            life = mf.get_lifetime(X, T, dt, halt_time, w=w, skip=skip)
            freqs.append(freq), lifes.append(life) 
        inds = np.arange(len(lifes))
        one = np.ones((len(lifes)))
        df = pd.DataFrame({'ind' :inds, 'life' : lifes, 
                           'k' : k* one, 'I' : I* one, 
                           'g' : g* one, 'f' : f* one, 
                           'p' : p * one}, 
                          index=np.arange(len(inds)))
        dfs.append(df)
        pd.concat(dfs).to_csv(outpath + '/summ_stats.csv')


