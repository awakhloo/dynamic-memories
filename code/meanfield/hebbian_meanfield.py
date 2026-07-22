import numpy as np 
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import argrelextrema, find_peaks
from tqdm import tqdm
# import torch
from glob import glob 
from collections import defaultdict

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController, Kvaerno3, Tsit5
import diffrax
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)



####### ##################

######### ODE solvers ###### #########

###########################

def halted_inps(t, thetas, I, omega, halt): 
    '''jax compatible halted oscillatory input''' 
    return jnp.where(t<halt,  I * jnp.cos(omega * t + thetas) , 0 )

def dxdt(t, x, J, A, omega, I, N, input_fn, phi=jnp.tanh):
    return (J+A)@phi(x)-x + input_fn(t)
    
def dadt(t,A, x, k, p, N,phi=jnp.tanh): 
    return 1/p * (k/N * jnp.outer(phi(x),phi(x)) - A)

def dfdt(t, Y, input_fn, N, J, omega, I, k, p, **kwargs): 
    ''''''
    x, A = Y[0:N], Y[N:N**2+N].reshape(N,N)
    xdiff = dxdt(t, x, J, A, omega, I, N, input_fn)
    adiff = dadt(t, A, x, k, p, N).reshape(N**2) 
    return jnp.concatenate([xdiff, adiff]) 

def solve_rand_system(sim_params, T, N, Nsave, rtol, atol, dt_eval, max_steps=12_000, A_times=None):
    ''' 
    the first arg is our sim params, and it should contain a dictioanry with anything that we could feasibly batch over
    '''
    # make the input func
    halt_t = sim_params['halt_time']*T
    inp_fn = lambda t: halted_inps(t, sim_params['thetas'], sim_params['I'], 
                                   sim_params['omega'], halt_t)
    # define the ODE and its solver params
    ode = lambda t, y, args: dfdt(t, y, inp_fn, N, **sim_params)
    ss= [diffrax.SubSaveAt(ts=np.linspace(0,T,int(T/dt_eval)), fn=lambda t,x,args : x[:Nsave])]
    if A_times is not None: 
        asave = diffrax.SubSaveAt(ts=A_times, fn=lambda t,x,args : x[N:].reshape(N,N),)
        ss.append(asave) 
    saveat = diffrax.SaveAt(subs=ss)
    term = ODETerm(ode)
    solver=Tsit5()
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    # solve
    sol = diffeqsolve(term, solver, t0=0, t1=T, dt0=0.1, y0=sim_params['Y0'], saveat=saveat,
                  stepsize_controller=stepsize_controller, max_steps=max_steps)
    return sol.ys

def solve_cust_system(sim_params, inp_fn, T, N, Nsave, rtol, atol, dt_eval, max_steps=12_000, A_times=None):
    ''' 
    allow for custom input functions. Note that this limits batching capabilities 
    '''
    # define the ODE and its solver params
    ode = lambda t, y, args: dfdt(t, y, inp_fn, N, **sim_params)
    ss= [diffrax.SubSaveAt(ts=np.linspace(0,T,int(T/dt_eval)), fn=lambda t,x,args : x[:Nsave])]
    if A_times is not None: 
        asave = diffrax.SubSaveAt(ts=A_times, fn=lambda t,x,args : x[N:].reshape(N,N),)
        ss.append(asave) 
    saveat = diffrax.SaveAt(subs=ss)
    term = ODETerm(ode)
    solver=Tsit5()
    stepsize_controller = PIDController(rtol=rtol, atol=atol)
    # solve
    sol = diffeqsolve(term, solver, t0=0, t1=T, dt0=0.1, y0=sim_params['Y0'], saveat=saveat,
                  stepsize_controller=stepsize_controller, max_steps=max_steps)
    return sol.ys


def solve_euler(sim_params,T, N, Nsave, rtol, atol, dt_eval, max_steps=12_000, dt=0.1):
    ''' 
    the first arg is our sim params, and it should contain a dictioanry with anything that we could feasibly batch over
    '''
    inp_fn = lambda t: halted_inps(t, sim_params['thetas'], sim_params['I'], 
                                   sim_params['omega'], sim_params['halt_time']*sim_params['T'])
    ode = lambda t, y, args: dfdt(t, y, inp_fn, N, **sim_params)
    Y = sim_params['Y0']
    Xs = [] 
    tvec = np.linspace(0,T,int(T/dt))
    for i in range(int(T/dt)):
        dy = ode(tvec[i],Y,0) 
        Y = Y + dt * dy
        Xs.append(Y[:N])
    return tvec, np.stack(Xs)


#######################################################

########## DATA PROCESSING ###########################

#############################################

def get_max_freq(Xfull, T, halt_time, dt, eps=1e-3, window_size=37.5): 
    ll=int(T*halt_time/dt)
    ul = int((T*halt_time + window_size)/dt)
    print(ll,ul)
    # ul=int(T*(halt_time+0.125)/dt)
    # ul=int(T*(halt_time+0.25)/dt)
    X = Xfull[ll:ul]
    psd = np.abs(fft(X, axis=0))**2 
    # average across units and batches 
    psd = psd.mean(1) 
    freqs = fftfreq(len(X), dt)
    # cut off past Nyquist Freq. 
    freqs, psd = freqs[:len(freqs)//2], psd[:len(psd)//2] 
    argpeaks=find_peaks(psd,threshold=eps)[0]
    if np.any(argpeaks):
        return freqs[argpeaks[0]] 
    else:
        return 0.  

def get_inst_freq(Xfull, T, halt_time, dt, eps=1e-3, window_size=37.5, step=1, use_full=False): 
    '''instantaneous frequency''' 
    if use_full is False: 
        ll=int(T*halt_time/dt)
        num_windows = 1 + int((T - T*halt_time - window_size)//step)
    else: 
        ll=0
        num_windows = int((T - window_size)//step)
    w = int(window_size/dt) # window size in time constant units 
    s = int(step/dt)
    windows = [(ll + s * i, ll + s * i + w) for i in range(num_windows)]
    assert windows[-1][-1] <= len(Xfull)
    ringing_freqs = [] 
    for (lower, upper) in windows: 
        X = Xfull[lower:upper]
        psd = np.abs(fft(X, axis=0))**2 
        # average across units and batches 
        psd = psd.mean(1) 
        freqs = fftfreq(len(X), dt)
        # cut off past Nyquist Freq. 
        freqs, psd = freqs[:len(freqs)//2], psd[:len(psd)//2] 
        argpeaks=find_peaks(psd,threshold=eps)[0]
        if np.any(argpeaks):
            ring_freq = freqs[argpeaks[0]] 
        else:
            ring_freq = 0.  
        ringing_freqs.append(ring_freq)
    tvec = np.linspace(-T*halt_time, T - T*halt_time, int(T/dt),endpoint=False)
    tvec = tvec[[w[0] for w in windows]] + window_size / 2 
    return np.array(ringing_freqs), tvec

def sliding_psd(X, window_size, skip):    
    # Calculate the power spectrum for each window
    power_spectra = []
    for i in range(0, X.shape[0] - window_size + 1, skip):
        windowed_signal = X[i:i+window_size]
        psds = np.abs(fft(windowed_signal,axis=0))**2
        psds = psds.mean(1) 
        psds = psds[:len(psds)//2]
        power_spectra.append(psds)
    return np.array(power_spectra)

def has_extrema(v,eps):
    return np.any(find_peaks(v, threshold=eps)[0])

def get_lifetime(X, T, dt, halt_time, w=400, skip=1,eps=1e-3): 
    Nh = int(T*halt_time/dt) 
    psds = sliding_psd(X[Nh:],w, skip)
    Tvec = np.linspace(T*halt_time, T, int(T*(1-halt_time)/dt/skip))
    extr = [has_extrema(psd,eps=eps) for psd in psds]
    # if there are no oscillations life= 0. 
    if extr[0] == False:
        return 0. 
    # otherwise get the index right before where a false first appears
    if np.isin(False, extr):
        ind = extr.index(False) - 1
        return Tvec[ind] 
    else:
        return Tvec[-1] 
        print('o')


####################################

######### ENTRAINMENT TRNST. #########
        
####################################
            

def is_fully_entrained(cov, tol, tol_m): 
    ''' 
    check for entrainment given a stationary covariance function
    '''
    var0 = cov.max() 
    # get the avg peak
    argmaxs = argrelextrema(cov, np.greater)[0]
    if len(argmaxs) <= 1: 
        return False 
    varmin = cov[argmaxs].min() 
    mean = cov.mean() 
    if np.abs(var0 - varmin) > tol or np.abs(mean) > tol_m: # enforce zero mean and a return to peak 
        return False
    else: 
        return True 


def autocov(x):
    ''' 
    Calculate the autocovariance of a time series. Assumes batch dim is the leading axis
    '''
    xhat = fft(x,axis=1)
    psd = (np.abs(xhat)**2).mean(0) 
    # phase shift 
    psd = np.array([np.exp(1j*np.pi*k) for k in range(x.shape[1])]) * psd 
    cov = ifft(psd).real
    cov = cov / x.shape[1] 
    return cov 



def get_crit_amp_sim(solv, params, N, T, dt, Nbatch, reps_bisec, I_lower, I_upper, tol, tol_m,init_time, burn, phi=np.tanh):
    ''' 
    Find the critical input amplitude of the network, given a fixed connectivity variance and input frequency. Use simulations for this 
    Args:
    - g: connectivity variance
    - I: input amplitude
    - f_lower: lower bound for the critical freq
    - f_upper: upper bound for critical freq
    '''
    I = (I_lower + I_upper) / 2 
    Is = [] 
    print('Burning = ', burn * dt) 
    for i in range(reps_bisec): 
        # draw a new set of params
        params['J'] = params['g'] / np.sqrt(N) *  np.random.randn(Nbatch,N,N)
        params['Y0'] = jnp.stack([jnp.concatenate([np.random.randn(N), np.zeros(N**2)]) for _ in range(Nbatch)])
        params['thetas'] = np.random.uniform(low=0,high=2*np.pi,size=(Nbatch,N,))
        # new I val 
        params['I'] = I 
        Xh, *_ = solv(params)  # batch x time x neu
        x = Xh[:, burn:]
        # covs = np.array([autocov(phi(sim[burn:, :, i].T)) for i in range(Nbatch)])
        covs = np.array([autocov(phi(Xh[i, burn:, :].T)) for i in range(Nbatch)])
        oscs = np.array([is_fully_entrained(cov, tol, tol_m) for cov in covs])
        print(I, oscs) 
        if np.mean(oscs) > 0.5:
            I_upper = I
        else: 
            I_lower = I
        # calculate new midpoint 
        I = (I_lower + I_upper) / 2  
        print('I = ', I,flush=True)
        Is.append(I) 
    return I, [I_lower, I_upper], Is, Xh



# def odeint(f, x0, t): 
#     ''' 
#     Integrate an ODE ∂x = f(x,t,X), where x is the value of x at time t, X is the vector of all previous values, and t is the time
#     '''
#     assert len(x0.shape) == 1 , 'only vectors'
#     x = x0 
#     x_all = np.zeros((len(t), len(x0))) 
#     dt = t[1] - t[0] 
#     for i, t in enumerate(t):
#         x_all[i] = x 
#         x = x + dt *  f(x,t,x_all[:i+1]) 
#     return x_all 



# #######

# # SIMULATION 

# #######

# def in_interval(t, intervals):
#     for intv in intervals:
#         if t >= intv[0] and t < intv[1]:
#             return True
#     return False

# def run_simulation_fast(N, T, T_eval, dt, g, k, p, I, omega, 
#                         N_batch, random_seed=None,  device='cuda',
#                         N_to_save=None, halt_intervals_syn=[], halt_intervals_osc=[],
#                         noise_intervals = [], sig=1.,
#                         connectivity=None, theta=None, x0=None):
    
#     ''' 
#     Run the hebbian networks simulation
#     Args:
#     - N: number of neurons
#     - T: time to simulate
#     - T_eval:  gap of time to evaluate over. Basically the 'dt' of the output array 
#     - g: variance of the connection 
#     - k: plasticity strenght
#     - p: plasticity timescale
#     - N_batch: batch dimension for the neurons 

#     '''
#     if N_to_save is None:
#         N_to_save = N
#     N_t = int(T / dt)
#     # N_K = N_t #int(7.5*p / dt) #keep track of states FACTOR*p in the past
#     N_K = int(10*p/dt)
#     Tvec = np.linspace(0, T, N_t, endpoint=False) 
#     if random_seed is not None:
#         np.random.seed(random_seed)
#     if connectivity is None: 
#         # couplings 
#         J = torch.randn(N, N, device=device)/np.sqrt(N)
#         perms1 = np.array([np.random.permutation(np.arange(N)) for _ in range(N_batch)]).T
#         perms2 = np.array([np.random.permutation(np.arange(N)) for _ in range(N_batch)]).T
#     else: 
#         J, perms1, perms2 = connectivity
        
#     # random phases
#     if theta is None:
#         theta = torch.tensor(np.random.uniform(low=0, high=2*np.pi, size=(N,N_batch)), device=device)
#     # history term
#     X = torch.zeros(N_K, N, N_batch, device=device)
#     # kernel 
#     K = torch.zeros(N_K, N_batch, device=device)
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#     # current state
#     if x0 is None:
#         x = torch.randn(N, N_batch, device=device)
#     else : 
#         x = x0.clone().detach()
#     X[0] = x
#     # exp ker
#     decaying_exp = torch.exp(-torch.arange(N_K, device=device)*dt/p)
#     if T_eval is not None:
#         eval_iter = int(T_eval / dt)
#         X_save = np.zeros((N_t // eval_iter, N_to_save, N_batch))#, dtype=np.float16)
#         X_save[0] = x[:N_to_save, :].cpu().numpy()#.astype(np.float16)
#     else:
#         X_save = None
    
#     for i in range(1, N_t):
#         integral_term = k*torch.trapz(torch.tanh(X)*K[:, None, :], dim=0, dx=dt)
#         phi = torch.tanh(x)
#         phip = phi[perms1, np.arange(N_batch)]
#         prod = torch.matmul(g*J, phip)
#         prodp = prod[perms2, np.arange(N_batch)]
#         ######
#         if not in_interval(i*dt, halt_intervals_osc):
#             osc_term = I*torch.cos(omega*Tvec[i] + theta)
#         else:
#             osc_term = 0 
#         if in_interval(i*dt, noise_intervals):
#             noise_term = sig*np.sqrt(dt)*torch.randn(N, N_batch, device=device)
#         else: 
#             noise_term = 0 
#         ######
#         dxdt = -x + prodp + integral_term + osc_term + noise_term 
#         x += dt*dxdt #update X variables
#         if not in_interval(i*dt, halt_intervals_syn):
#             X = torch.roll(X, 1, dims=(0,))
#             X[0] = x
#         #now need to update kernel...
#         Phi_decay = torch.tanh(X)*decaying_exp[:, None, None]
#         K = (1./p)*(Phi_decay*torch.tanh(x)[None, :, :]).sum(dim=1)/N
#         if T_eval is not None and i % eval_iter == 0:
#             X_save[i//eval_iter] = x[:N_to_save, :].cpu().numpy()#.astype(np.float16)
#             norms = np.linalg.norm(x.cpu().numpy(), axis=0)/np.sqrt(N)
#             if np.max(norms) < 1e-2:
#                 print("norm small -- breaking!")
#                 break
#     return X_save



# #######

# # COVARIANCE UTILITIES 

# #######

# def autocov(x):
#     ''' 
#     Calculate the autocovariance of a time series. Assumes batch dim is the leading axis
#     '''
#     xhat = fft(x,axis=1)
#     psd = (np.abs(xhat)**2).mean(0) 
#     # phase shift 
#     psd = np.array([np.exp(1j*np.pi*k) for k in range(x.shape[1])]) * psd 
#     cov = ifft(psd).real
#     cov = cov / x.shape[1] 
#     return cov 


# def autocov_brute(x): 
#     '''
#     Calculate autocovariance "by hand"  
#     '''
#     covs = np.zeros((x.shape[0], x.shape[1]) ) 
#     Tmax = x.shape[1]
#     for j in range(x.shape[1]): 
#         covs[:,j] = 1/(Tmax-j) * np.sum(x[:, j:] * x[:, :Tmax-j], axis=1)
#     cov = covs.mean(0) 
#     return np.concatenate([np.flip(cov), cov])#[::2] 



# ##################

# ###### Find phase transition point using a bisection search 

# ##################


# def is_oscillating(cov, tol): 
#     ''' 
#     Check if a covariance indicates that a network is oscillating 
#     ''' 
#     var0 = cov.max() 
#     # get the avg peak
#     argmaxs = argrelextrema(cov, np.greater)[0]
#     print(cov[argmaxs])
#     if len(argmaxs) <= 1: # if there's only one peak it's not oscillating
#         return False 
#     varmin = cov[argmaxs].min() 
#     if np.abs(var0 - varmin) > tol: # if there is a gap between peaks then it's not oscillating
#         return False
#     else: 
#         return True 
    

# def is_fully_entrained(cov, tol, tol_m): 
#     ''' 
#     check for entrainment given a stationary covariance function
#     '''
#     var0 = cov.max() 
#     # get the avg peak
#     argmaxs = argrelextrema(cov, np.greater)[0]
#     if len(argmaxs) <= 1: 
#         return False 
#     varmin = cov[argmaxs].min() 
#     mean = cov.mean() 
#     if np.abs(var0 - varmin) > tol or np.abs(mean) > tol_m: # enforce zero mean and a return to peak 
#         return False
#     else: 
#         return True 

         


# def get_crit_amp_sim(N, T, T_eval, dt, g, k, p, omega, N_batch, reps_bisec, I_lower, I_upper, tol, tol_m,init_time, burn=None, device='cuda', phi=np.tanh):
#     ''' 
#     Find the critical input amplitude of the network, given a fixed connectivity variance and input frequency. Use simulations for this 
#     Args:
#     - g: connectivity variance
#     - I: input amplitude
#     - f_lower: lower bound for the critical freq
#     - f_upper: upper bound for critical freq
#     '''
#     I = (I_lower + I_upper) / 2 
#     Is = [] 
#     halt_intervals_osc = [[0, init_time * T]]
#     if burn is None: 
#         burn = int(T/dt/2)
#     print('Burning = ', burn * dt) 
#     for i in range(reps_bisec): 
#         oscs =[] 
#         sim = run_simulation_fast(N, T, T_eval, dt, g, k, p, I, omega, 
#                     N_batch, random_seed=None,  device=device,
#                     N_to_save=None, halt_intervals_syn=[], halt_intervals_osc=halt_intervals_osc)
#         x = sim[burn:]
#         print(is_fully_entrained(autocov(x[...,0].T),1e-3,1e-1))
#         covs = np.array([autocov(phi(sim[burn:, :, i].T)) for i in range(N_batch)])
#         oscs = np.array([is_fully_entrained(cov, tol, tol_m) for cov in covs])
#         print(oscs)
#         if np.mean(oscs) > 0.5:
#             I_upper = I
#         else: 
#             I_lower = I
#         # calculate new midpoint 
#         I = (I_lower + I_upper) / 2  
#         print('I = ', I,flush=True)
#         Is.append(I) 
#     return I, [I_lower, I_upper], Is, sim


# #####

# ##### RINGING UTILS

# ######


# # def sliding_psd(X, window_size, skip=1):    
# #     # Calculate the power spectrum for each window
# #     power_spectra = []
# #     for i in range(0, X.shape[0] - window_size + 1, skip):
# #         windowed_signal = X[i:i+window_size]
# #         psds = np.abs(fft(windowed_signal,axis=0))**2
# #         psds = psds.mean(1) 
# #         psds = psds[:len(psds)//2]
# #         power_spectra.append(psds)
# #     return np.array(power_spectra)

# # def has_extrema(v):
# #     return np.any(argrelextrema(v,np.greater))


# # import time

# # def get_lifetime(X, T, dt, halt_time, w=400, subsample=500, skip=1): 
# #     '''expect X with shape time x neurons''' 
# #     X = X[:, :subsample]
# #     t0 = time.time() 
# #     psds = sliding_psd(X,w,skip=skip) 
# #     Tvec = np.linspace(0,T,int(T/dt/skip))
# #     t1 = time.time() 
# #     extr = [has_extrema(psd) for psd in psds]
# #     # if there are no oscillations life= 0. 
# #     if not np.isin(True, extr):
# #         return 0. 
# #     # otherwise get the index right before where a false first appears
# #     if np.isin(False, extr):
# #         ind = extr.index(False) - 1
# #         # if this is before the input is halted, return 0. 
# #         if ind < int(T*halt_time/dt):
# #             return 0.
# #         else:
# #             return Tvec[ind] 
# #     else:
# #         return Tvec[-1] 


# # def is_ringing(X, T, halt_time, dt, window, phi=np.tanh, gap=None): 
# #     if len(X.shape) == 2: 
# #         X = X[..., None] 
# #     nbatch = X.shape[2] 
# #     rings = [] 
# #     t_thresh = T * (halt_time + gap)
# #     for n in range(nbatch):
# #         Xb = X[...,n]
# #         life = get_lifetime(Xb, T, dt, halt_time, w=window)
# #         rings.append(life > t_thresh) 
# #     print('vals = ', rings) 
# #     return np.mean(rings)


# def get_ringing_amp(sim_fn, I_upper, I_lower,reps_bisec, T, halt_time, dt, window, mode, phi=np.tanh, gap=None):
#     I = (I_lower + I_upper) / 2 
#     Is = [] 
#     assert mode in ['below', 'above'], 'must be either below or above'
#     for i in range(reps_bisec): 
#         x = sim_fn(I) 
#         if is_ringing(x, T, halt_time, dt, gap=gap, phi=phi,window=window):
#             if mode == 'below':
#                 I_upper = I
#             elif mode == 'above':
#                 I_lower = I
#         else:
#             if mode == 'below': 
#                 I_lower = I
#             elif mode == 'above': 
#                 I_upper = I
#         # calculate new midpoint 
#         I = (I_lower + I_upper) / 2  
#         print('I = ', I,flush=True)
#         Is.append(I) 
#     return I, [I_lower, I_upper], Is

# def get_ringing_amp_sim(N, T, T_eval, dt, g, k, p, omega, N_batch, reps_bisec, I_lower, I_upper, I_ringing, halt_time, init_time, device='cuda', phi=np.tanh, gap=None, window=5):
#     ''' 
#     Get the upper and lower boundaries of the ringing phase where the ringing phase is defined as having a peak in its PSD 
#     Args:
#     -N: number of neurons to simulate
#     -T: time of the interval to sim over
#     -T_eval: time step to record from
#     -dt: time step for the simulation
#     -g: variance
#     -k: synapse strength
#     -p: synaptic timescale
#     -omega: input freq
#     -N_batch: number of networks to simulate in parallel
#     -reps_bisec: number of times to iterate bisection on either side
#     -I_lower: input amp I<I_ringing s.t. the network is not ringing in this state
#     -I_upper: input amp I>I_ringing s.t. the network is not ringing in this state
#     -I_ringing: input amp that leaves the network ringing
#     -halt_time: fraction of the interval to let the network run before halting the input
#     -gap: fraction of the total interval to wait before checking for the ringing (i.e., we calculate PSD from T*(gap + halt_time) until the end 
#     -window: size of smoothing window to use i nthe calculation of PSD 
#     -device: device to run 
#     -phi: nonlinearity 
#     '''
#     halt_intervals_osc = [[0, init_time * T], [T * halt_time, np.inf]]
#     sim_fn = lambda I : run_simulation_fast(N, T, T_eval, dt, g, k, p, I, omega, 
#                                         N_batch, random_seed=None,  device=device,
#                                         N_to_save=None, halt_intervals_syn=[], 
#                                         halt_intervals_osc=halt_intervals_osc)
#     I_L, *_ = get_ringing_amp(sim_fn, I_ringing, I_lower, reps_bisec, T, halt_time, dt, window,
#                               mode='below', phi=phi, gap=gap)
#     I_U, *_ = get_ringing_amp(sim_fn, I_upper, I_L, reps_bisec, T, halt_time, dt, window,
#                               mode='above', phi=phi, gap=gap)
#     return I_L, I_U
    
    

# #######

# ##### TARGET EIGENVECTORS 


# ######


# def in_interval(t, intervals):
#     for intv in intervals:
#         if t >= intv[0] and t < intv[1]:
#             return True
#     return False

# def find_edge_eigval_idx(targ, eig, eps=0.05):
#     within_angle_mask = torch.abs(targ - eig.imag) < eps
#     good_idxs = torch.arange(len(eig))[within_angle_mask]
#     best_idx = good_idxs[eig.real[good_idxs].argmax()]
#     return best_idx

# def find_eig_idx(targ_im, targ_re, eig): 
#     return np.argmin(np.abs(eig - targ_re - 1j * targ_im))

# def sim_custom_osc(N, T, T_eval, dt, k, p,  
#                         J,osc_fn, random_seed=None,  device='cuda',
#                         N_to_save=None, halt_intervals_syn=[], halt_intervals_osc=[],
#                         noise_intervals = [], sig=1., A_init=None):
    
#     ''' 
#     Run the hebbian networks simulation
#     Args:
#     - N: number of neurons
#     - T: time to simulate
#     - T_eval:  gap of time to evaluate over. Basically the 'dt' of the output array 
#     - g: variance of the connection 
#     - k: plasticity strenght
#     - p: plasticity timescale
#     - N_batch: batch dimension for the neurons 
#     - J: a tensor of shape (N, N, N_batch) 
#     - osc_fn: external input. (t -> (N, N_batch) 
#     '''
#     if N_to_save is None:
#         N_to_save = N
#     N_t = int(T / dt)
#     N_K = int(7.5*p / dt) #keep track of states FACTOR*p in the past
#     N_batch = J.shape[-1]
#     Tvec = np.linspace(0, T, N_t, endpoint=False) 
#     # random phases
#     theta = torch.tensor(np.random.uniform(low=0, high=2*np.pi, size=(N,N_batch)), device=device)
#     # history term
#     X = torch.zeros(N_K, N, N_batch, device=device)
#     # kernel 
#     K = torch.zeros(N_K, N_batch, device=device)
#     if random_seed is not None:
#         torch.manual_seed(random_seed)
#     # current state
#     x = torch.randn(N, N_batch, device=device)
#     X[0] = x
#     # exp ker
#     decaying_exp = torch.exp(-torch.arange(N_K, device=device)*dt/p)
#     if T_eval is not None:
#         eval_iter = int(T_eval / dt)
#         X_save = np.zeros((N_t // eval_iter, N_to_save, N_batch))#, dtype=np.float16)
#         X_save[0] = x[:N_to_save, :].cpu().numpy()#.astype(np.float16)
#     else:
#         X_save = None
#     if random_seed is not None:
#         np.random.seed(random_seed)
    
#     for i in range(1, N_t):
#         # history self-coupling
#         integral_term = k*torch.trapz(torch.tanh(X)*K[:, None, :], dim=0, dx=dt)
#         phi = torch.tanh(x)
#         # jphi term
#         prodp = torch.einsum('ijb, jb -> ib', J, phi)
#         # term [1/p e^{-t/p} A(0)] phi
#         if A_init is not None:
#             init_A_term = torch.einsum('ijb, jb -> ib', torch.exp(-torch.tensor(Tvec[i])/p) * A_init, phi) 
#         else:
#             init_A_term = 0.
#         ######
#         if not in_interval(i*dt, halt_intervals_osc):
#             osc_term = osc_fn(Tvec[i]) 
#         else:
#             osc_term = 0 
#         if in_interval(i*dt, noise_intervals):
#             noise_term = sig*np.sqrt(dt)*torch.randn(N, N_batch, device=device)
#         else: 
#             noise_term = 0 
#         ######
#         dxdt = -x + prodp + integral_term + osc_term + noise_term + init_A_term 
#         x += dt*dxdt #update X variables
#         if not in_interval(i*dt, halt_intervals_syn):
#             X = torch.roll(X, 1, dims=(0,))
#             X[0] = x
#         #now need to update kernel...
#         Phi_decay = torch.tanh(X)*decaying_exp[:, None, None]
#         K = (1./p)*(Phi_decay*torch.tanh(x)[None, :, :]).sum(dim=1)/N
#         if T_eval is not None and i % eval_iter == 0:
#             X_save[i//eval_iter] = x[:N_to_save, :].cpu().numpy()#.astype(np.float16)
#             norms = np.linalg.norm(x.cpu().numpy(), axis=0)/np.sqrt(N)
#             if np.max(norms) < 1e-2:
#                 print("norm small -- breaking!")
#                 break
#     return X_save


# def find_edge_eigval_idx(targ, eig, eps):
#     within_angle_mask = torch.abs(targ - eig.imag) < eps
#     good_idxs = torch.arange(len(eig))[within_angle_mask]
#     best_idx = good_idxs[eig.real[good_idxs].argmax()]
#     return best_idx



# ### DATA PROCESSING UTILS
# ## maximum frequency
# def get_max_freq(Xfull, T, halt_time, dt, eps=1e-8, window_size=37.5): 
#     ll=int(T*halt_time/dt)
#     ul = int((T*halt_time + window_size)/dt)
#     # ul=int(T*(halt_time+0.125)/dt)
#     # ul=int(T*(halt_time+0.25)/dt)
#     X = Xfull[ll:ul]
#     psd = np.abs(fft(X, axis=0))**2 
#     # average across units and batches 
#     psd = psd.mean(1) 
#     freqs = fftfreq(len(X), dt)
#     # cut off past Nyquist Freq. 
#     freqs, psd = freqs[:len(freqs)//2], psd[:len(psd)//2] 
#     argpeaks=find_peaks(psd,threshold=eps)[0]
#     if np.any(argpeaks):
#         return freqs[argpeaks[0]] 
#     else:
#         return 0.  

# def sliding_psd(X, window_size, skip):    
#     # Calculate the power spectrum for each window
#     power_spectra = []
#     for i in range(0, X.shape[0] - window_size + 1, skip):
#         windowed_signal = X[i:i+window_size]
#         psds = np.abs(fft(windowed_signal,axis=0))**2
#         psds = psds.mean(1) 
#         psds = psds[:len(psds)//2]
#         power_spectra.append(psds)
#     return np.array(power_spectra)

# def has_extrema(v,eps=1e-8):
#     return np.any(find_peaks(v, threshold=eps)[0])

# def get_lifetime(X, T, dt, halt_time, w=400, skip=5,eps=1e-8): 
#     Nh = int(T*halt_time/dt) 
#     psds = sliding_psd(X[Nh:],w, skip)
#     Tvec = np.linspace(T*halt_time, T, int(T*(1-halt_time)/dt/skip))
#     extr = [has_extrema(psd,eps=eps) for psd in psds]
#     # if there are no oscillations life= 0. 
#     if extr[0] == False:
#         return 0. 
#     # otherwise get the index right before where a false first appears
#     if np.isin(False, extr):
#         ind = extr.index(False) - 1
#         return Tvec[ind] 
#     else:
#         return Tvec[-1] 
#         print('o')


# def get_A(X, T, dt, target_times,k,p): 
#     ''' 
#     reconstruct A matrix from the activity traces
#     '''
#     N_t, N, N_batch = X.shape
#     Tvec = np.linspace(0,T,int(T/dt),endpoint=False)
#     A_all = np.zeros((len(target_times), N, N, N_batch)) 
#     Phi = torch.tanh(X)
#     A = 0 
#     j=0
#     for i in range(N_t): 
#         if Tvec[i] in target_times: 
#             A_all[j] = A 
#             j += 1 
#         PhiPhit = torch.einsum('ib, jb -> ijb', Phi[i], Phi[i]).cpu().numpy()
#         dAdt = (-A + k/N * PhiPhit) / p 
#         A = A + dt * dAdt
#     return A_all 

        
# def get_dfs(path):
#     files = glob(path + '/k*.npy')
#     res = defaultdict(dict) 
#     dfs = []
#     for j, f in tqdm(enumerate(files)): 
#         arr = np.load(f, allow_pickle=True).item()
#         halt_time, init_time, T, dt_sim = arr['halt_time'], arr['init_time'], arr['T'], arr['T_eval']
#         dt_actual = arr['dt'] 
#         k, g, p, f, I = arr['k'], arr['g'], arr['p'], arr['f'], arr['I']
#         _, NSave, Nbatch = arr['sim_X'].shape
#         N,p,I,k,f = arr['N'], arr['p'], arr['I'], arr['k'], arr['f']
#         for i in range(arr['sim_X'].shape[-1]): 
#             if 'freqs' in arr.keys(): 
#                 omega = arr['freqs'][i]
#             else:
#                 omega = mf.get_max_freq(arr['sim_X'][...,i],T, halt_time, dt_sim)
#             if 'life' in arr.keys(): 
#                 life = arr['life'][i]
#             else:
#                 life = mf.get_lifetime(arr['sim_X'][...,i], T, dt_sim, halt_time, w=400)
#             life = np.maximum(life - T * halt_time, 0.)
#             df = pd.DataFrame({'N' : N, 'p' : p, 'k' : k, 'I': I, 'f' : f, 'om_max' : omega, 'life' : life},
#                               index=[0])
#             dfs.append(df)
#         dfs.append(df)

#     df = pd.concat(dfs, ignore_index=True)
#     df['dt'] = dt_actual
#     # mdf = df.groupby(['N', 'p', 'k', 'I', 'f','dt']).median().reset_index()
#     mdf = df.groupby(['N', 'p', 'k', 'I', 'f','dt']).mean().reset_index()
#     sdf = df.groupby(['N', 'p', 'k', 'I', 'f','dt']).std().reset_index()
#     return df, mdf, sdf