
''' 
make fig 5. 
'''

import sys
import os 
from tqdm import tqdm
sys.path.append(os.getcwd()) 

import numpy as np 
import pandas as pd
import meanfield.hebbian_meanfield as mf
from importlib import reload
import jax.numpy as jnp 
reload(mf)

# np.random.seed(23423563)
np.random.seed(64209)
outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'

k =3.0
p = 25.
f = 0.1
I = 6.
g = 1.3 
omega = f * 2 * np.pi 


# time params 
T = 700
dt = 0.1
halt_time = 2/7
halt = T * halt_time
# sim params
N = 500
Nsave = N
Nbatch = 5000



outpath = outdir + f'/test_every_eig/test_every_eig_g_{g}_T_{T}_dt_{dt}_I_{I}_mult'
os.makedirs(outpath, exist_ok=True) 



# window for lifetime 
w=400
eps=1e-2
skip=1

# solver params
rtol, atol=1e-6, 1e-6



params = {
          'g' : g, 'I' : I, 'f' : f, 'k' : k, 
          'p' : p, 'omega' : omega,
          'T' : T, 'halt_time' : halt_time, 
        'halt' : T * halt_time,
         }

dfs = [] 
for K, b in enumerate(range(Nbatch)): 
    params['J'] = g/np.sqrt(N) * np.random.randn(N,N)
    params['Y0'] = jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) 
    params['thetas'] = np.random.uniform(low=0,high=2*np.pi,size=(N,))
    ind = np.random.choice(np.arange(N))
    eigs, evecs = np.linalg.eig(params['J'])
    eig = eigs[ind] 
    e = evecs[:,ind]
    u, v = e.real, e.imag 
    def targ_inp(t): 
        out= I * np.sqrt(N) * (u * jnp.cos(omega*t) + v * jnp.sin(omega*t))
        out_mask = jnp.where(t<halt,  out, 0 )
        return out_mask
    Xh, *_ = mf.solve_cust_system(params, targ_inp, T=T, N=N, Nsave=Nsave, rtol=rtol,
                              atol=atol, dt_eval=dt)
    life = mf.get_lifetime(Xh, T, dt, halt_time, w=w, skip=skip, eps=eps)
    freq = mf.get_max_freq(Xh,  T, halt_time,dt)
    df = pd.DataFrame({'eig_r' : eig.real, 'eig_i' : eig.imag, 
          'life' : life, 'freq' : freq,
          'N' :  N, 'f' : f, 'g' : g, 'I' : I, 'k' : k, 'p' : p, 'dt' : dt, 
          'T' : T, 'halt_time' : halt_time}, index=[ind])
    dfs.append(df)
    if (K+1) % 10 == 0:
        pd.concat(dfs).to_csv(outpath + f'/results_N_{N}.csv')
    

## DEMO 
np.random.seed(7532)
targ = 1 + 0.8j
print(np.abs(targ))
n_sub = 5 # num batch to save
A_times = np.linspace(0, T, int(T))
all_eigs = [] 

for b in range(n_sub):
    params['J'] = g/np.sqrt(N) * np.random.randn(N,N)
    params['Y0'] = jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) 
    params['thetas'] = np.random.uniform(low=0,high=2*np.pi,size=(N,))
    eigs, evecs = np.linalg.eig(params['J'])
    ind = np.argmin(np.abs(eigs - targ))
    e = evecs[:,ind]
    u, v = e.real, e.imag 
    def targ_inp(t): 
        out= I * np.sqrt(N) * (u * jnp.cos(omega*t) + v * jnp.sin(omega*t))
        out_mask = jnp.where(t<halt,  out, 0 )
        return out_mask
    Xh, A = mf.solve_cust_system(params, targ_inp, T=T, N=N, Nsave=Nsave, rtol=rtol,
                              atol=atol, dt_eval=dt, A_times=A_times)
    eigs = jnp.linalg.eigvals(params['J'] + A)
    all_eigs.append(eigs) 
        
all_eigs = np.stack(all_eigs)

np.save(outpath + '/demo_results.npy', 
        {'eigs' : all_eigs, 'target_times' : A_times, 'target' : targ})






