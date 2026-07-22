



######## 

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
import gc
jax.config.update("jax_enable_x64", True)

from importlib import reload; reload(mf)

outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'


ks = [3.0] 
Is = [1.0] 
ps = [15.]
fs = [0.1]
g = 1.3 

# ps = np.linspace(10,20,11) 
# mod = 'p' 

Is = np.linspace(0, 1.5, 21)
mod = 'I'


# time params 
T = 800
dt = 0.1
halt_time = 0.1875
init_time = 0.0

# long time network params
T_long = 800
init_time_long = 0.1 
downscale_time = 0.3
halt_time_long = 0.6 
big_scale = 5. # scaling of amplitude during "breakout" period


# sim params
Ns = [1000]
Nsave = 500 
Nsavenp =  15 
Nrep = 1
Nbatch = 200

outdir = outdir + f'/compare_initialization_g_{g}_T_{T}_dt_{dt}/mod_{mod}_sim'
os.makedirs(outdir, exist_ok=True) 

# window
w = 400
skip=50
relu = lambda x : x*(x>0)

# solver 
rtol = 1e-6 
atol = 1e-6

def osc_long(t, thetas, I, omega):
    out = 0 
    out = out + jnp.where((t >= init_time_long*T) & (t < downscale_time*T), big_scale * jnp.cos(omega * t + thetas) , 0 )
    out = out + jnp.where((t >= downscale_time*T) & (t < halt_time_long*T), I * jnp.cos(omega * t + thetas) , 0 )
    return out 

def get_results(Xh, T, halt_time,dt): 
    freqs, lifes = [], [] 
    for b in range(Nbatch):
        X = Xh[b]
        freq = mf.get_max_freq(X, T, halt_time, dt)
        life = mf.get_lifetime(X, T, dt, halt_time, w=w,skip=skip)
        life = relu(life - T * halt_time)
        freqs.append(freq), lifes.append(life) 
    results = {'f' : f, 'k' : k, 'p' : p, 'I' : I, 'g' : g,
                'T' : T, 'dt' : dt,  'halt_time' : halt_time, 
                'init_time' : init_time, 'Xh' : np.array(Xh[..., :Nsavenp]), 
               'life' : np.array(lifes), 'freqs' : np.array(freqs), 
               'N' : N}
    return results 
    

# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}


for d, N in enumerate(Ns):
    # short and long exposures
    sol_short = lambda params : mf.solve_rand_system(params,
                                                   T=T, N=N, Nsave=Nsave, rtol=rtol,atol=atol,dt_eval=dt)
    SOL_short = jit(vmap(sol_short, in_axes=(mapped_inds,)))

    sol_lon = lambda params : mf.solve_cust_system(params,
                                                   lambda t: osc_long(t, 
                                                                      params['thetas'],
                                                                      params['I'],
                                                                      params['omega']),
                                                   T=T, N=N, Nsave=Nsave, rtol=rtol,atol=atol,dt_eval=dt)
    SOL_lon = jit(vmap(sol_lon, in_axes=(mapped_inds,)))
    for rep in range(Nrep): 
        for a, k in enumerate(ks):
            for c, I in tqdm(enumerate(Is)): 
                for b, p in enumerate(ps):
                    for e, f in enumerate(fs):
                        omega = 2*np.pi*f
                        batched_params ={'J' : g/np.sqrt(N) * jnp.array(np.random.randn(Nbatch,N,N)), 
                                      'Y0' : jnp.stack([jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) for _ in range(Nbatch)]), 
                                      'thetas' : jnp.array(np.random.uniform(low=0,high=2*np.pi,size=(Nbatch,N,))),
                                      'g' : g, 'I' : I, 'f' : f, 'k' : k, 
                                      'p' : p, 'omega' : omega,
                                      'T' : T, 'halt_time' : halt_time, 
                                    'halt' : T * halt_time,
                                     }
                        Xs, *_ = SOL_short(batched_params) 
                        Xl, *_ = SOL_lon(batched_params) 
                        res_short, res_lon = get_results(Xs,T,halt_time,dt), get_results(Xl,T,halt_time_long,dt)
                        results = {'short' : res_short, 'long' : res_lon}
                        np.save(outdir + f'/k_{k}_g_{g}_I_{I}_p_{p}_N_{N}_f_{f}_comparison_rep_{rep}.npy', np.array(results))
                        gc.collect() 






