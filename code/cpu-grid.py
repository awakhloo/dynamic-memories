

import sys
import os 
from tqdm import tqdm
sys.path.append(os.getcwd()) 

import numpy as np 
import pandas as pd
import meanfield.hebbian_meanfield as mf
from importlib import reload
import torch 
import time
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
reload(mf)

device='cuda'
outdir = '/mnt/home/awakhloo/ceph/persistent-oscillations/results'
np.random.seed(6432)
torch.manual_seed(936562)



# p = 15. 
p = 25 
g = 1.3
f = 0.1 
omega = 2 * np.pi * f
ks = np.linspace(0, 3.5, 61)
Is = np.linspace(0, 2.0, 61)
# ks = np.linspace(0, 3.5, 21)
# Is = np.linspace(0, 2.0, 21)




device = 'cuda' 

# time params 
T =  800  # 500 #
dt = 0.1
T_eval = dt
halt_time =  3/8 #1/4 #0.5
init_time = 0.0
tvec = np.linspace(0,T,int(T/dt))
tvec_eval = np.linspace(0,T,int(T/T_eval))

# intervals
halt_intervals_osc = [[T * halt_time, np.inf]]
halt_intervals_syn = [] 
sig = 3.0
noise_intervals = [] 

# sim params
N = 250 
N_batch = 500

# N = 1000
# N_batch = 50

# N = 500 
# N_batch = 10 


# lifetime params
w=400
skip = 50 # +/- 2.5s lifetime measure

dfs = []

# parameters to make spot check figure with smaller dt and over a smaller grid 
spot_check = False
prefix = ''
if spot_check: 
    dt=0.01
    T_eval = 0.1
    T = 1000 
    halt_time = 1/2 
    ks = np.linspace(1.5, 3.5,5)
    Is = np.linspace(0.25, 1.5,6)
    prefix = 'spot_check_'


print(prefix) 
outpath = outdir + f'/{prefix}grid/grid_N_{N}_dt_{dt}_w_{w}_T_eval_{T_eval}_T_{T}'
print(outpath,flush=True) 
os.makedirs(outpath,exist_ok=True)

np.random.seed(23423)
seeds = np.random.randint(low=0,high=10_000,size=(len(Is), len(ks), N_batch))


def dxdt(t, x, J, thetas, A, omega, I, N, halt, phi):
    if t<halt:  
        return (J+A)@phi(x)-x + I * np.cos(omega * t + thetas) 
    else: 
        return (J+A)@phi(x)-x
    
def dadt(t,A, x, k, p, N,phi): 
    return 1/p * (k/N * np.outer(phi(x),phi(x)) - A)

def dfdt(t,Y, J, thetas, omega, I, k, p, N, halt,phi): 
    '''
    x, A = Y[:N], Y[N:].reshape(N,N)
    xdiff = dxdt(t, x, J, thetas, A, omega, I, N, halt,phi)
    adiff = dadt(t,A,x,k,p,N,phi).reshape(-1) 
    return np.concatenate([xdiff, adiff])   


def one_iter(seed): 
    '''draw the random params and run the model on a single batch''' 
    np.random.seed(seed)
    J = g / np.sqrt(N) * np.random.randn(N,N)
    thetas = np.random.uniform(low=0,high=2*np.pi,size=(N,))
    x0, A0 = np.random.randn(N) / np.sqrt(N), np.zeros(N**2) 
    Y0 = np.concatenate([x0,A0]) 
    ode = lambda t, Y: dfdt(t,Y,J,thetas,omega,I,k,p,N,halt,phi)

    y = Y0
    y_eval = [y] 
    for i in range(1,len(tvec)): 
        y = y + dt * ode(tvec[i-1],y) 
        if tvec[i] in tvec_eval: 
            y_eval.append(y)
    return np.stack(y_eval)
            
    

for i, k in enumerate(ks):
    for j, I in tqdm(enumerate(Is)):
        L = []
        for n in range(N_batch): 
            X = one_iter(seeds[i,j,n]) 
            life = mf.get_lifetime(X[...,n], T, T_eval,halt_time, w=w, skip=skip)
            L.append(life)
        inds = np.arange(len(L))
        one = np.ones((len(L)))
        df = pd.DataFrame({'ind' :inds, 'life' : L, 
                           'k' : k* one, 'I' : I* one, 
                           'g' : g* one, 'f' : f* one, 
                           'p' : p * one}, 
                          index=np.arange(len(inds)))
        dfs.append(df)
        pd.concat(dfs).to_csv(outpath + '/summ_stats.csv')




#### SAVE A FEW RUNS FOR THE PHASE DIAGRAM 



def run_example_sim(k, I, n_seed, t_seed, f, p, g, N, 
                    halt_time=halt_time, init_time = init_time, T=T):
    omega = f * 2 * np.pi
    N_batch=5 # N_batch=1
    N_to_save = 15  
    dt=0.1
    Tvec = np.linspace(0,T,int(T/dt), endpoint=False)
    T_eval = np.maximum(0.01, dt)
    
    halt_intervals_osc = [[0, init_time * T], [T * halt_time, np.inf]]
    halt_intervals_syn = [] 
    sig = 0.
    noise_intervals = [] 

    device = 'cuda'
    np.random.seed(n_seed) 
    J = np.random.randn(N, N)/np.sqrt(N)
    J = torch.tensor(J, device=device)
    perms1 = np.array([np.random.permutation(np.arange(N)) for _ in range(N_batch)]).T
    perms2 = np.array([np.random.permutation(np.arange(N)) for _ in range(N_batch)]).T
    connectivity = (J, perms1, perms2) 

    sim_args = {'N' : N, 'T' : T, 'T_eval' : T_eval, 'dt' : dt,
                'g' : g, 'k' : k, 'p' : p, 'I' : I, 'omega' : omega,
                'N_batch' : N_batch, 'device' : 'cuda', 'N_to_save' : N_to_save,
                'connectivity': connectivity,
               'halt_intervals_osc' : halt_intervals_osc, 
                'halt_intervals_syn' : halt_intervals_syn, 
                'noise_intervals' : noise_intervals, 
                'sig' : sig, 'random_seed' : t_seed
               }
    X2 = mf.run_simulation_fast(**sim_args)
    return X2.squeeze()

ks = [0.3, 2.5, 3.0, 3.0]
Is = [0.25, 0.25, 0.55, 1.9] 
# f, p, g, N = 0.1, 15, 1.3, 5_000 
# n_seeds = [1334634, 23423124, 63546, 93446756] 
# t_seeds = [23861, 58219475, 84345, 53582]
n_seeds = [1334634, 23423124, 463546, 93446756] 
t_seeds = [23861, 58219475, 184345, 53582]
Xs = [] 

for k, I, n_seed, t_seed in tqdm(zip(ks, Is, n_seeds, t_seeds)): 
    X = run_example_sim(k, I, n_seed, t_seed, f=f, p=p, g=g, N=N) 
    Xs.append(X)
Xs = np.stack(Xs, axis=0) 
out = outdir + '/grid/example_sim.npy'
np.save(out, Xs)
