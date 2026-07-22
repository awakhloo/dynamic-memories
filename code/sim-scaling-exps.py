


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


np.random.seed(974324)


# default params
ks = [3.0] 
Is = [.75] 
# ps = [15.]
ps = [25.]
fs = [0.1]
g = 1.3 

# sim params

Ns = [5_000] 
Nbatch = 10
nrep = 3 # how many runs w this batch size
Nsave = 500 # how many to save from simul
Nsavenp = 30 # how many to actually save


# fast
# Ns = [1000] 
# Nbatch = 20
# nrep = 1 # how many runs w this batch size
# Nsave = 500 # how many to save from simul
# Nsavenp = 30 # how many to actually save

# huge N 
# Ns = [10000] 
# Nbatch = 1
# nrep = 30 # how many runs w this batch size
# Nsave = 500 # how many to save from simul
# Nsavenp = 30 # how many to actually save

# small n 
# Ns = [1000] 
# Nbatch = 200
# nrep = 2 # how many runs w this batch size
# Nsave = 500 # how many to save from simul
# Nsavenp = 30 # how many to actually save



# time params 
T = 700
dt = 0.1
halt_time = 2/7
init_time = 0.0
time_params = [(T, halt_time)]

# solver 
rtol = 1e-6 
atol = 1e-6



## CONFIG

# ps = np.linspace(10, 30,21)
# Is = [0.5, 0.75, 1.0]
# mod = 'p' 

# Is = np.linspace(0, 1.5, 21)
# ps = [10., 15., 20., 25.]
# mod = 'I'

# fs = np.linspace(0.01, 0.16,21)
# Is = [0.5, 0.75, 1.0]
# mod = 'f'


Is = [0.65]
ps = [15, 20, 25]
ks = [3.0]
halt_ts = np.linspace(25, 300, 12) #np.linspace(25, 400, 12)
delay = 600
time_params = [(ht + delay, ht/(delay+ht)) for ht in halt_ts]
print(time_params, flush=True)
mod = 'runuptime'


outpath = outdir + f'/scaling_sims/g_{g}_dt_{dt}_scale_{mod}_sim'
os.makedirs(outpath, exist_ok=True) 

# window
w = 400
skip = 50

# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}

for (T, halt_time) in time_params:
    for d, N in enumerate(Ns):
        SLV = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol, atol=atol, dt_eval=dt)
        solv = jit(vmap(SLV, in_axes=(mapped_inds,)))
        for rep in range(nrep): 
            for a, k in enumerate(ks):
                for c, I in tqdm(enumerate(Is)): 
                    for b, p in enumerate(ps):
                        for e, f in enumerate(fs):
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
                            t2 = time.time() 
                            print("sim time = ", t1-t0)
                            print("life calc time = ", t2-t1) 
                            print('\n')
                            results = {'f' : f, 'k' : k, 'p' : p, 'I' : I, 'g' : g,
                                        'T' : T, 'dt' : dt, 'halt_time' : halt_time, 
                                        'init_time' : init_time, 'Xh' : np.array(Xh[..., :Nsavenp]), 
                                       'life' : np.array(lifes), 'freqs' : np.array(freqs), 
                                       'N' : N} 
                            np.save(outpath + f'/k_{round(k,6)}_g_{round(g,6)}_I_{round(I,6)}_p_{round(p,6)}_N_{N}_f_{round(f,6)}_T_{round(T,6)}_halt_{round(halt_time,6)}_rep_{rep}_comparison.npy', np.array(results))
                            gc.collect()



