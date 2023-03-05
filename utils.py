import os,glob,csv,scipy
from scipy.io import wavfile
import numpy as np

from IPython import display
import matplotlib.pyplot as plt

"""
filter_lpc : filter structure
---------------------------------
         1
    ____________
      1 + PolA
---------------------------------
A : coeffficientes of the polynomial
rate : sampling rate of the signal's file
"""
class filter_lpc:
    def __init__(self,A,rate=48000):
        self.size = len(A)
        self.A = np.array(A).reshape((self.size,1))
        self.rate = rate
    
    # calculates filter prediction
    # returns result, error of the prediction
    def evaluate(self,wav):
        inputs = []
        for i in range(self.size,len(wav)+1):
            inputs += [wav[i-self.size:i]]
        inputs = np.matrix(inputs)
        result = np.array(inputs*self.A)[:,0]
        return result,np.array(wav)[:-self.size]-result[:-1]
    
    # returns spectrum after the filter
    def spectrum(self,freqs):
        filtered_freq = []
        for i in freqs:
            S =0
            x= np.exp(np.array(-1j*2*np.pi*i/self.rate))
            for u in range(self.size):
                S+= self.A[u][0]*x**(u+1)
            filtered_freq += [20*np.log(abs(1/(1-S)))]
        return np.array(filtered_freq)
    
    #returns roots of the filter
    def roots(self):
        roots = np.roots(list(-np.array(self.A)[:,0])[::-1]+[1])
        ampli = abs(roots)
        arg = np.angle(roots)
        freq = arg*self.rate/2/np.pi
        return freq,ampli,arg

    #returns a simulation of the signal with the filter
    def simu(self,init,iters=100):
        sig = list(init)
        for i in range(iters):
            result = np.array(np.matrix(sig[-self.size:])*self.A)[:,0]
            sig += [float(result)]
        return sig
    
    #returns formants of the original signal
    def formants(self):
        freq,a,arg = self.roots()
        freq = freq[::-1]
        formants = []
        acc = []
        for fi in freq:
            if(len(acc) == 0):
                acc += [float(fi)]
            else:
                if(acc[0] == -float(fi)):
                    formants += [abs(acc[0])]
                acc = [float(fi)]
        return set(formants)
        
"""
LPC : calculates filter coefficientes to fit of the signal envelope
    wav : the temporal signal
    rate : sampling rate of the signal's file
    p : the degree of the filter + 1
return filter
""" 
def LPC(wav,rate=48000,p=10):
    wav = np.array(wav)
    n = len(wav)
    #calculate correlation
    corr =[float(np.correlate(wav,wav))/n]
    
    for i in range(1,p+1):
        corr += [float(np.correlate(wav[:-i],wav[i:]))/n]
    
    #calculate coef
    R = []
    
    line = corr[:-1]
    r = np.array(corr[1:]).reshape((p,1))
    for i in range(p):
        R+= [line[p-i:][::-1]+line[:p-i]]
    
    R = np.matrix(R)
    A = np.linalg.inv(R)* r
    f = filter_lpc(list(A),rate)
    return f

"""
harmo : calculates harmonics of the first fundamental frequency
    spectre : amplitudes
    freq : frequencies
    nb_harm = number of harmonics
return amplitudes of harmonics , frequencies of the harmonics
"""
def harmo(spectre,freq,nb_harm=15):
    funda = 0
    for i in range(len(spectre)):
        if(spectre[i]>spectre[funda]):
            funda = i
        elif(spectre[i]<spectre[funda]/100 and(funda >20 and  funda*3/2<i)):
            break;
    x_harm =[]
    L_harm =[]
    for i in range(nb_harm):
        ind = int(np.rint((i+1)*freq[funda]/freq[1]))
        max_id =ind-i
        for j in range(ind-i,ind+i):
            if(spectre[j]>spectre[max_id]):
                max_id = j
        L_harm += [spectre[max_id]]
        x_harm += [freq[max_id]]
    return L_harm,x_harm


"""
#****EXAMPLE****
os.chdir('C:/Users/Admin/Documents/Python Scripts/project voice/DATA/vowel')

#wav_file_name = 'Mes chers compatriotes des Antilles.wav'
wav_file_name = 'E_Rod_8000.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

nb = 500
offset=10000
cut = nb//2
p = 8
W = np.array(wav_data[offset:offset+nb])
#t =np.array([2*np.pi*i/sample_rate for i in range(nb) ])
#W = np.sin(1000*t) + 0.1*np.sin(500*t) + 0.05*np.sin(1500*t)
#x = [0 for i in range(nb//4)]+x_init+[0 for i in range(nb//4)]
ham_win = np.hamming(len(W))
x = W*ham_win
f = LPC(x,p=p,rate=sample_rate)
s,e = f.evaluate(x)
ax = np.linspace(0.0,sample_rate, len(W))
fft = abs(np.fft.fft(W))

ax = ax[:cut]

sp = f.spectrum(ax)


efft = np.log(abs(np.fft.fft(e)))[:cut]
fft = fft[:cut]
lfft = np.log(fft)
res = np.log(fft)*sp

plt.plot(ax,sp)
"""
