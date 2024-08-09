# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:02:50 2023

@author: Admin
"""

from utils  import LPC
import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


#****EXAMPLE****
os.chdir('C:/Users/Admin/Documents/Python Scripts/project voice/DATA/Mozilla/cv-corpus-12.0-delta-2022-12-07/fr/clips')

#wav_file_name = 'Mes chers compatriotes des Antilles.wav'
wav_file_name = 'common_voice_fr_34929481.wav'
sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

nb = 500
offset=100000
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
plt.show()