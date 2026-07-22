import numpy as np 
import matplotlib.pyplot as plt 
import scipy.special as sp 
from scipy.integrate import dblquad, odeint
from scipy.optimize import fsolve
from scipy.fft import fft, ifft, rfft
from scipy.signal import correlate, argrelextrema
from scipy.linalg import dft
from tqdm import tqdm
import meanfield.hebbian_meanfield as mf
from importlib import reload

import torch.distributions as dist
import torch

reload(mf)
import time

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)

def plot_fancy(X,ax,color, init_time=0., halt_time=0.5, T=300, dt=0.1, Nplot=6, fn=14, lw=2, draw_inputs=True): 
    T = np.linspace(0,T,int(T/dt), endpoint=False)
    alphas = np.linspace(0.2,1.0,Nplot)
    for i in range(Nplot):
        ax.plot(T, X[:, i], color=color,alpha=alphas[i], lw=lw)
    if draw_inputs:
        ymin, ymax = ax.get_ylim()
        ax.vlines(T[-1] * halt_time, ymin, ymax, color='grey', ls=':', lw=2, label='halt input')
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'$t$', fontsize=fn) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax 


def plot_traces(X2, N_batch, T, dt, I, k, p, f, g, N, halt_time, init_time, liml=None, limu=None):
    ''' 
    plot neural traces. note that halt_time and init_time denote the fraction of the itnerval of the event and as such should be between [0,1] 
    '''
    Tvec = np.linspace(0, T, int(T/dt), endpoint=False) 
    # fig, axs = plt.subplots(2,3, figsize=(22,10))
    fig, axs = plt.subplots(N_batch,1, figsize=(6,3.5*N_batch))
    N_t = T/dt 
    fn =10
    # fn = 2 * N_batch 
    if liml is None:
        liml = 0 # int(N_t * 1/8)
    if limu is None:
        limu = int(N_t) # int(3*N_t//4) #int(N_t) #int(N_t) #i

    for j, ax in enumerate(axs.flat):
        for i in range(5):
            ax.plot(Tvec[liml:limu], X2[liml:limu,i,j], lw=2.)
        ax.axvline(T * halt_time, ls = '--', label = 'halt', color='red', lw=2.)
        ax.axvline(T * init_time, ls ='--', label='init', color='black', lw=2.)
        ax.set_title(f"I = {I}, k = {k}, p = {p}, f = {f}, g = {g}, N = {N}, dt={dt}", fontsize=fn)
        ax.set_xlabel('$t$', fontsize=fn) 
        ax.set_ylabel('$x(t)$', fontsize=fn)
        ax.legend(fontsize=fn) 
    plt.tight_layout()
    return axs
    
    
def plot_psd(X2, N_batch, T, dt, I, k, p, f, g, N, halt_time, init_time, liml=None, limu=None):
    ''' 
    plots the psd of the neural autocovariance after halting the input 
    '''
    Tvec = np.linspace(0, T, int(T/dt), endpoint=False) 
    fn = 14
    if liml is None:
        liml = int(T*halt_time/dt)
    if limu is None: 
        limu = int(100*T/dt) 
    slce = X2[liml:limu]
    psds = np.abs(fft(slce, axis=0))**2
    psds = psds.mean((1,))
    N_samples = slce.shape[0]

    fs = 1/dt 
    nyquist_freq = fs / 2
    freq_resolution = fs / N_samples
    freqs = np.arange(0, nyquist_freq, freq_resolution)
    fig, ax = plt.subplots()
    for j in range(psds.shape[1]):
        ax.plot(freqs, psds[:len(freqs),j])
    ax.set_title(f"I = {I}, k = {k}, p = {p}, f = {f}, g = {g}, N = {N}, dt = {dt}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(f,ls='--', label='input freq')
    ax.legend(fontsize=fn * 2/3)
    ax.set_xlabel('$f$', fontsize=fn)
    ax.set_ylabel('$|\hat x(\omega)|^2$', fontsize=fn)
    plt.tight_layout()
    
    # fig, axs = plt.subplots(int(N_batch/2),2, figsize=(2.5*N_batch,10/6*N_batch))
    # for j, ax in enumerate(axs.flat):
    #     ax.plot(freqs, psds[:len(freqs),j])
    #     ax.set_title(f"I = {I}, k = {k}, p = {p}, f = {f}, g = {g}, N = {N}")
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.axvline(f,ls='--', label='input freq')
    #     ax.legend(fontsize=fn * 2/3)
    #     ax.set_xlabel('$f$', fontsize=fn)
    #     ax.set_ylabel('$|\hat x(\omega)|^2$', fontsize=fn)
    # plt.tight_layout()
    return psds 
    