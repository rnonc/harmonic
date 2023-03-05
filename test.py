import os,glob,scipy
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import opensmile
from scipy import signal
from tools import exctract, split_window, fond, complex_harmonic_resolver, filt_real_fond, Formant, roots_poly, LPC_coef, del_init_zero

def test_fft():
    # file name without extension
    #****EXAMPLE****
    #folder = 'C://Users/Admin/Documents/Python Scripts/project voice/DATA/Mozilla/cv-corpus-12.0-delta-2022-12-07/fr/clips'
    #os.chdir(folder)


    #wav_file_name = 'Mes chers compatriotes des Antilles.wav'
    #wav_file_name = 'common_voice_fr_36530166.wav'
    #sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

    #fft = np.fft.fft(wav_data)[:int(len(wav_data)/2)]
    sample_rate = 32000#8000
    sig = [np.sin(2*np.pi*200*i/sample_rate+3) + np.sin(2*np.pi*400*i/sample_rate+1) + np.sin(2*np.pi*600*i/sample_rate)+0.001*(np.random.rand((1))-0.5) for i in range(300)]


    ex = sig#wav_data[18000:24000]#wav_data[55000:56000]

    cut = 30

    
    f = [90+1*i for i in range(2500)]


    result = exctract(ex,f,sample_rate)

    plt.plot(f,abs(result)*2/len(ex),label = 'fft')
    plt.plot(sample_rate/len(ex)*np.linspace(0,cut-1,num=cut),abs(np.fft.fft(ex)[0:cut])*2/len(ex),label = 'discrete fft')
    harmo = 10
    freq = 200
    fft_result = exctract(ex,[freq*i for i in range(1,harmo+1)],sample_rate)/len(ex)*2
    alpha = complex_harmonic_resolver(ex,freq/sample_rate,harmo,real=True,zero=False)
    plt.plot([freq*i for i in range(1,harmo+1)],abs(alpha),'+',label = 'harmonic resolver solutions with 200Hz')
    plt.plot([freq*i for i in range(1,harmo+1)],abs(fft_result),'+',label = 'fft solutions with 200Hz')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency calculate with 1000 time steps')
    plt.legend()
    print(abs(fft_result))
    print(abs(alpha))


def test_smile():
    # file name without extension
    #****EXAMPLE****
    folder = 'C://Users/Admin/Documents/Python Scripts/project voice/DATA/Mozilla/cv-corpus-12.0-delta-2022-12-07/fr/clips'
    os.chdir(folder)
    
    
    #wav_file_name = 'Mes chers compatriotes des Antilles.wav'
    wav_file_name = 'common_voice_fr_36530167.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    
    # Extact this for our test sentence, out comes a pandas dataframe
    result_df = smile.process_file(wav_file_name)
    
    # Now us only the three center formant frequencies
    centerformantfreqs = ['F1frequency_sma3nz', 'F2frequency_sma3nz', 'F3frequency_sma3nz']
    formant_df = result_df[centerformantfreqs]
    a,b = result_df.shape
    ax = [ i*len(wav_data)/a for i in range(a)]
    #plt.plot(ax,list(result_df['F1frequency_sma3nz']))
    #plt.plot(ax,list(result_df['F2frequency_sma3nz']))
    #plt.plot(ax,list(result_df['F3frequency_sma3nz']))
    #plt.plot(wav_data)
    #plt.plot(ax,list(np.array(result_df['alphaRatio_sma3'])*200))
        
    M,_ = split_window(wav_data,size_window = 300,step=1)
    M = np.log(np.transpose(abs(np.fft.fft(M))))
    
    plt.pcolormesh(M)
    plt.plot(ax,list(np.array(result_df['alphaRatio_sma3'])*0.1+5))

def test_formant_exctraction():
    folder = 'C://Users/Admin/Documents/Python Scripts/project voice/DATA/Mozilla/cv-corpus-12.0-delta-2022-12-07/fr/clips'
    #folder = 'C:/Users/Admin/Documents/Python Scripts/project voice/DATA/vowel'
    os.chdir(folder)


    #wav_file_name = 'Mes chers compatriotes des Antilles.wav'
    wav_file_name = 'common_voice_fr_36530166.wav'
    #wav_file_name = 'E_Rod_8000.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

    data = del_init_zero(wav_data)[::4][2000:10000]
    sample_rate = sample_rate/4
    win_data,x = split_window(data,size_window=100,step=75)
    Coef = LPC_coef(win_data,p=11)
    Roots = roots_poly(Coef)
    freq,ampli = Formant(Roots,sample_rate)

    plt.plot(data,label= 'signal')
    plt.plot(x,freq,label=['Formant 1','Formant 2','Formant 3'])
    plt.legend()
    plt.xlabel('time step')
    plt.ylabel('signal amplitude / Formant frequency')

def test_harmonic_exctraction():
    folder = 'C://Users/Admin/Documents/Python Scripts/project voice/DATA/Mozilla/cv-corpus-12.0-delta-2022-12-07/fr/clips'
    os.chdir(folder)


    wav_file_name = 'common_voice_fr_36530166.wav'
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

    sig = np.array([10*np.cos(2*np.pi*210*i/sample_rate) + np.cos(2*np.pi*420*i/sample_rate+3) 
                    + np.cos(2*np.pi*630*i/sample_rate) + np.cos(2*np.pi*840*i/sample_rate) +0.001*(np.random.rand((1))-0.5) for i in range(2000)])


    #sig = sig.reshape((1,sig.shape[0]))
    sig = sig.reshape((sig.shape[0],))
    ex = np.array(wav_data[2000:10000])#wav_data[55000:56000]




    M,x = split_window(ex,200,50)
    fo = fond(M,sample_rate)
    print(fo)
    harmo = 10
    alpha = np.zeros((M.shape[0],harmo),dtype=complex)
    for i in range(M.shape[0]):
        if(fo[i] is None):
            alpha[i] = np.array([None for i in range(harmo)])
        else:
            alpha[i] = complex_harmonic_resolver(M[i],fo[i]/sample_rate,harmo,real=True,zero=False).reshape((harmo,))
    plt.plot(x,abs(alpha)[:,:4],label = ['Harmonic ' + str(i) for i in range(4)])
    plt.legend(loc = 1)
    plt.xlabel('time step')
    plt.ylabel('harmonic amplitudes')


