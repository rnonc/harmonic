# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:04:05 2023

@author: Rodolphe Nonclercq
"""
import os,glob,scipy
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import opensmile
from scipy import signal

def exctract(sig,freq,sample_rate):
    vect = -2j*np.pi/sample_rate*np.linspace(0,len(sig)-1,num=len(sig)).reshape((1,len(sig)))
    
    freq = np.array(freq).reshape((len(freq),1))
    table = np.exp(np.matmul(freq,vect))
    return np.matmul(table,np.array(sig).reshape((len(sig),1)))


def harmonic_matrix(theta,n_harmonic,n_data,zero=False,neg=False):
    if(zero):
        expo = np.matmul(np.linspace(0,n_data-1,num=n_data).reshape(n_data,1),np.linspace(0,n_harmonic,num=n_harmonic+1).reshape(1,n_harmonic+1))
        
    else:
        expo = np.matmul(np.linspace(0,n_data-1,num=n_data).reshape(n_data,1),np.linspace(1,n_harmonic,num=n_harmonic).reshape(1,n_harmonic))
    if(neg):
        result = np.exp(-2j*np.pi*theta*expo)
    else:
        result = np.exp(2j*np.pi*theta*expo)
    return result

"""
theta = freq_fonda/freq_e
"""
def complex_harmonic_resolver(sig,theta,n_harmonic=20,real=False,zero=False,proj=True):
    if(len(sig)<n_harmonic):
        print("less data, need more than {n_harmonic} points")
        return None
    if not real :
        M = np.matrix(harmonic_matrix(theta,n_harmonic,len(sig),zero=zero))
    else:
        MD = np.matrix(harmonic_matrix(theta,n_harmonic,len(sig),zero=False))
        MG = np.matrix(harmonic_matrix(theta,n_harmonic,len(sig),zero=False,neg=True))[:,::-1]
        if zero:
            M = np.concatenate((MG,1+np.zeros((len(sig),1)),MD),axis=1)
        else:
            M = np.concatenate((MG,MD),axis=1)
    M_t = np.transpose(M)
    MM_inv = np.linalg.inv(M_t*M)
    b = np.matmul(M_t,np.array(sig).reshape(len(sig),1))
    
    #print(np.linspace(0,b.shape[0]-1,num=b.shape[0]).reshape((1,b.shape[0]))*b.reshape((1,b.shape[0])))
    #print(2*abs(b[n_harmonic:])/len(sig))
    a = np.matmul(MM_inv,b)
    if real and proj:
        a *=2
        a = a[n_harmonic:]
    return a


def split_window(sig,size_window=500,step = 1):
    M = np.zeros((int((len(sig)-size_window)/step),size_window))
    x = []
    for i in range(int((len(sig)-size_window)/step)):
        M[i] = sig[step*i:step*i+size_window]
        x.append(step*i+size_window/2)
    return M,x

def autocorr_Mat(M,i=0):
    return np.sum(M[:,i:]*M[:,:-i],axis=1)

def LPC_coef(M,p=10,window='Gauss'):
    if(window == 'Gauss'):
        window = np.array(signal.windows.gaussian(M.shape[1], std=7))
        M = window*M
    corr  = np.zeros((M.shape[0],p+1))
    for i in range(1,p+2):
        corr[:,i-1] = autocorr_Mat(M,i)
    
    line = corr[:,:-1]
    R = np.zeros((M.shape[0],p,p))
    for i in range(p):
        R[:,i] = np.concatenate((line[:,p-i:][:,::-1],line[:,:p-i]),axis=1)
    r = corr.reshape((corr.shape[0],p+1,1))[:,1:]
    
    A = np.concatenate((1+np.zeros((M.shape[0],1)),-np.matmul(np.linalg.inv(R), r).reshape((R.shape[0],R.shape[1]))),axis=1)
    return A

def roots_poly(Coef):
    Roots = np.zeros((Coef.shape[0],Coef.shape[1]-1),dtype=complex)
    for i in range(Coef.shape[0]):
        Roots[i] = np.roots(Coef[i][::-1])
    return Roots

def Formant(roots,rate=32000):
    arg = np.angle(roots)
    ampli = abs(roots)
    freq = arg*rate/2/np.pi
    freq_Formant = np.zeros((roots.shape[0],3))
    ampli_Formant = np.zeros((roots.shape[0],3))
    for elem,line in enumerate(freq):
        L = []
        X = []
        for x,i in enumerate(line):
            if(i != 0 and -i in line and not abs(i) in L):
                L.append(abs(i))
                X.append(ampli[elem,x])
        if(len(L)<3):
            freq_Formant[elem] = [None,None,None]
            ampli_Formant[elem] = [None,None,None]
        else:
            for i in range(3):
                x = np.argmin(L)
                freq_Formant[elem,i] = L[x]
                ampli_Formant[elem,i] = X[x]
                L.pop(x)
                X.pop(x)
        if freq_Formant[elem,0] >800:
            freq_Formant[elem] = [None,None,None]
            ampli_Formant[elem] = [None,None,None]
    
    return freq_Formant,ampli_Formant

def del_init_zero(L):
    i = 0
    j = -1
    while(i<len(L) and L[i] == 0):
        i += 1
    while(-j<len(L) and L[i] == 0):
        j -= 1
    return L[i:j]

def filt_real_fond(fond):
    if fond >70 and fond < 410:
        return True
    else:
        return False

"""
exctract fundamental frequency
"""
def fond(M,sample_rate,filt = lambda x: True):
    D = np.zeros((M.shape[1],M.shape[1]))
    X = np.zeros((M.shape[0],M.shape[1]))
    D[0,0] = -1
    D[0,1] = 1
    D[-1,-1] = 1
    D[-1,-2] = -1
    for i in range(1,D.shape[0]-1):
        D[i,i] = -2
        D[i,i-1] = 1
        D[i,i+1] = 1
    D = D
    X= np.transpose(np.matmul(D,np.transpose(M)))
    freq = np.sqrt(-np.sum(M[:,1:-1]*X[:,1:-1],axis=1)/np.sum(M[:,1:-1]*M[:,1:-1],axis=1))/2/np.pi*sample_rate
    for i,f in enumerate(freq):
        if not filt(f):
            freq[i] = None
            
    return freq
