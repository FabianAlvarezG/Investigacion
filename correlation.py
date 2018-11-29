# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:46:43 2018

@author: Asus
"""
import numpy as np
import scipy.integrate 
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

### Desplaza la funcion u en l
## Solo funciona si dx es 1!
def shift(u_actual, x, l, dx=1):
    L = len(x)
    xx = np.arange(x[0], x[0]+2*L,dx)
    xxx = np.arange(x[0]-L, x[0]+L,dx)
    uu = np.concatenate([u_actual, u_actual])
    if l >=0:
        u = interp1d(xx, uu , kind='cubic')
    else:
        u = interp1d(xxx, uu , kind='cubic')
    x_new = x + l
    return u(x_new)

def correlation(vector, x, l):
    L = len(x)
    vector_new = shift(vector, x, l)
    return scipy.integrate.simps(np.conjugate(vector)*vector_new,x)/L

def model_exp(x, A, r0):
    return A * np.exp(-x/r0)

def corr_function(vector, x):
    L = np.arange(0,len(x)/4,.05)
    corr = []
    for l in L:
        c = correlation(vector,x,l)
        corr.append(c)
    corr = np.array(corr)
    return corr
        

def corr_length(vector,x):
    correla = corr_function(vector,x)
    corr = correla[1]
    L = correla[0]
    return curve_fit(model_exp, L, np.abs(corr))[0][1]
    
def plot_corr(vector, x):
    correla = corr_function(vector,x)
    corr = correla[1]
    L = correla[0]
    A, r0 = (curve_fit(model_exp, L, np.abs(corr))[0][0], curve_fit(model_exp, L, np.abs(corr))[0][1])
    plt.plot(L, np.abs(corr), '.')
    plt.plot(L, model_exp(L, A, r0))
    plt.show()
    


def correlation_kinchin(vector):
    fourier = np.fft.fft(np.abs(vector)**2)
    return np.abs(fourier)[70:len(vector)/4]

def corr_length_kinchin(vector):
    corr = correlation_kinchin(vector)
    L = np.arange(0,len(corr))
    return curve_fit(model_exp, L, np.abs(corr))[0][1]
    
def plot_corr_kinchin(vector):
    corr = correlation_kinchin(vector)
    L = np.arange(0,len(corr))
    A, r0 = (curve_fit(model_exp, L, np.abs(corr))[0][0], curve_fit(model_exp, L, np.abs(corr))[0][1])
    plt.semilogy(L, np.abs(corr), '.')
    plt.semilogy(L, model_exp(L, A, r0))
    plt.show()

    
    
   

     