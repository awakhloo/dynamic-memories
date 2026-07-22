

'''
fig 4 
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

from importlib import reload; reload(mf)

outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'

np.random.seed(974324)

config = 'imlambdafreq' 


if config == 'dense' : 
    # Is = [0.75, 0.25, 1.25]
    # ks = [2.5, 3.0, 3.5] 
    # fs = [0.075, 0.1, .125] 
    T = 700 
    dt = 0.1
    halt_time = 2/7
    Is = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 2.25, 2.5]
    ks = [2.5, 3.0, 3.5] 
    fs = [0.075, 0.1, 0.15]
    ps = [25.0] 
    g = 1.3
    N = 500
    Nbatch=50
    # N = 1000 
    # Nbatch = 75
    Nsavenp=15
    A_times = np.array([T*halt_time])
    print('evaluating at ', A_times,flush=True) 

elif config == 'examples' : 
    # T = 300 
    dt=0.1
    # T = 700 
    # halt_time = 2/7
    T = 500 
    halt_time = 2/5 
    Is = [0.6, 0.25] #[0.65, 0.25]
    ks = [3.0]
    fs = [0.1] 
    ps = [25.0] 
    g = 1.3
    N = 500
    Nbatch=5
    Nsavenp=15
    A_times = np.linspace(0, T, int(T/dt/10), endpoint=False)


elif config == 'runuptime' : 
    T = 500 
    dt=0.1
    halt_time = 0.8
    init_time = 0.0
    Is = [0.75]
    ks = [3.0]
    fs = [0.1] 
    ps = [ 15, 20, 25, 30] 
    g = 1.3
    N = 500
    Nbatch=15
    Nsavenp=15
    A_times = np.linspace(0, T, int(T/dt/10), endpoint=False)


elif config == 'imlambdafreq':
    T = 700 
    dt=0.1
    halt_time = 2/7
    init_time = 0.0
    Is = [0.75, 1.25]
    ks = [2.5, 3.0]
    fs = [0.1, 0.15] 
    ps = [20] 
    g = 1.3
    N = 500
    Nbatch=5
    Nsavenp=500
    A_times = np.linspace(0, T, int(T/dt/10), endpoint=False)
    


# stats calc
Nsave= 500 
w = 400 # window for calculating a sliding PSD, the peaks of which are used for determining lifetime
skip = 1 
# solver 
rtol = 1e-6 
atol = 1e-6



outpath=outdir+f'/eig_at_halt/A-matrices-dt-{dt}_grid_{config}_N_{N}'
os.makedirs(outpath, exist_ok=True)

# indices that we'll vmap over
mapped_inds = {'J' : 0, 'Y0' : 0, 'thetas' : 0, 
               'k' : None, 'f' : None, 'omega' : None, 'I' : None, 
               'p' : None, 'g' : None, 'T' : None, 
               'halt_time' : None, 'halt' : None}

SLV = lambda params : mf.solve_rand_system(params, T=T, N=N, Nsave=Nsave, rtol=rtol, atol=atol, dt_eval=dt, A_times=A_times)
solv = jit(vmap(SLV, in_axes=(mapped_inds,)))

for f in fs: 
    for k in ks: 
        for p in ps: 
            for I in tqdm(Is): 
                
                t0 = time.time() 
                omega = 2*np.pi*f
                batched_params ={'J' : g/np.sqrt(N) * jnp.array(np.random.randn(Nbatch,N,N)), 
                                      'Y0' : jnp.stack([jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) for _ in range(Nbatch)]), 
                                      'thetas' : jnp.array(np.random.uniform(low=0,high=2*np.pi,size=(Nbatch,N,))),
                                      'g' : g, 'I' : I, 'f' : f, 'k' : k, 
                                      'p' : p, 'omega' : omega,
                                      'T' : T, 'halt_time' : halt_time, 
                                    'halt' : T * halt_time,
                                     }
                Xh, A = solv(batched_params) # (Batch, time, neuron)
                
                t1 = time.time() 
                print("sim time = ", t1-t0, flush=True) 
                
                # get the stats 
                freqs, lifes = [], [] 
                for b in range(Nbatch):
                    X = Xh[b]
                    freq = mf.get_max_freq(X, T, halt_time, dt)
                    life = mf.get_lifetime(X, T, dt, halt_time, w=w)
                    freqs.append(freq), lifes.append(life) 
                t2 = time.time() 
                print("life calc time = ", t2-t1, flush=True) 
                    
                W = batched_params['J'][:, None] + A # A is (batch,time,n,n) 
                eig = [] 
                for t in tqdm(range(W.shape[1])):
                    E = jnp.linalg.eigvals(W[:,t])
                    eig.append(E)
                res = {'I' : I, 'g' : g, 'k' : k, 'p' : p,
                       'halt_time' : halt_time, 'N' :N, 'f' : f, 
                       'T' : T, 'dt' : dt, 'A_times' : A_times, 
                       'eigs' : np.stack(eig), 'Xh' : Xh[...,:Nsavenp], 
                       'life' : np.array(lifes), 'freqs' : np.array(freqs)
                      }
                if config == 'examples' : 
                    res['J'] = np.array(batched_params['J'])
                    res['A'] = np.array(A)
                np.save(outpath + f'/eigvals_p_{p}_I_{I}_f_{f}_g_{g}_k_{k}.npy', np.array(res))
                gc.collect() 