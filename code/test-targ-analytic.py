
''' 
test the eqn for the targeted outlier
'''

import sys
import os 
from tqdm import tqdm
sys.path.append(os.getcwd()) 

import numpy as np 
import jax.numpy as jnp
import pandas as pd
import meanfield.hebbian_meanfield as mf
from importlib import reload
reload(mf)


np.random.seed(85356)
outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'

# time params 
T = 600
dt = 0.1
halt_time = 1.0
init_time = 0.0
A_times = np.array([T])

# param grid
ks = np.linspace(2., 5., 10) 
targs = np.concatenate([0.8 + np.linspace(0.3, 0.8, 10) * 1j,
                        1.0 + np.linspace(0.3, 0.8, 10) * 1j])
print([np.abs(t) for t in targs])
ps = [25] 
fs = [0.1, 0.15, 0.05] 
gs = [1.3] 
Is = [30, 15] 

# neuron params
phi = jnp.tanh
N = 500
N_to_save=None
Nbatch =1

# solver
atol, rtol =1e-6, 1e-6
Nsave = 15



outpath = outdir + f'/test_every_eig/test_targ_analytic'
os.makedirs(outpath, exist_ok=True) 

assert Nbatch==1, 'code is only written for one batch'


dfs= [] 
for I in Is:
    for g in gs: 
        for f in fs: 
            for p in ps: 
                for targ in targs:
                    for k in tqdm(ks): 
                        # draw params
                        omega =  f * 2 *np.pi
                        params = {'I' : I, 'g' : g, 'f' : f, 'omega' : omega, 'p' : p, 'k' : k,
                                  'halt' : T*halt_time, 'halt_time' : halt_time, 'T' : T}
                        J = g/np.sqrt(N) * np.random.randn(N,N)
                        eig, evecs = jnp.linalg.eig(J)
                        ind = np.argmin(np.abs(targ - eig))
                        eig_targ = eig[ind] 
                        e = evecs[:,ind]
                        u, v = e.real, e.imag 
                        def targ_inp(t): 
                            return I * np.sqrt(N) * (u * jnp.cos(omega*t) + v * jnp.sin(omega*t))
                        params['J'] = J
                        params['Y0'] = jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) 
                        params['thetas'] = np.random.uniform(low=0,high=2*np.pi,size=(N,))
                        # solve dynamics 
                        Xh, A = mf.solve_cust_system(params, targ_inp, T=T, N=N, Nsave=Nsave, rtol=rtol,
                                                  atol=atol, dt_eval=dt, A_times=A_times)
                        # get the outlier and save 
                        eigs = jnp.linalg.eigvals(params['J'] + A.squeeze())
                        outlier = eigs[np.argmax(np.abs(eigs))]
                        results = {'I' : I, 
                                    'g' : g, 'f' : f,
                                    'p' : p, 'k' :k, 'targ' : targ,
                                   'targ_emp' : eig_targ, 
                                    'eig_out' : outlier}
                        df = pd.DataFrame(results, index=[0]) 
                        dfs.append(df)
                        pd.concat(dfs).to_csv(outpath + f'/results.csv')
                    