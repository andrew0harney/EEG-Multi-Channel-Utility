import numpy as np
import matplotlib.pyplot as plt
import tables
import pandas as pd
from andyClasses.NewBrain import NewBrain
from andyClasses.gridFT import calcFFT, plot_morlet,morlet
import pickle 

fs = 2000
ws = 0.1
x = np.linspace(-np.pi, np.pi,fs)
times = np.arange(0,2,1.0/fs)

sig1 = np.hstack([np.sin(50*x),np.sin(50*x)+np.sin(90*x)+np.sin(30*x)])
sig2 = np.hstack([np.sin(90*x),np.sin(40*x)+np.sin(90*x)+np.sin(70*x)])
sig = np.array([sig1,sig2])


dataName= 'PeriodicMorlet'
NewBrain.save_hdf(sig, times, ['s1','s2'],base_file_name=dataName)
events = np.vstack((times,times+0.1,np.zeros(len(times)))).T
events = pd.DataFrame(events,columns=['pulse.on','pulse.off','event.code'])
events.to_csv(dataName+'Test_events',columns=['pulse.on','pulse.off','event.code'])

grid = NewBrain(dataName,log_file=dataName+'Test_events',fs=2000)
grid.set_wd(['s1','s2'])
blocks = pd.DataFrame(np.array([[1,1.9995],[1,1.9995]]),columns=['pulse.on','pulse.off'])
isi = pd.DataFrame(np.array([[0,0.9995],[0,0.9995]]),columns=['pulse.on','pulse.off'])


ws = 0.1
dec = 1
frequencies = np.arange(1,101,1)

norms = grid.normSignal(frequencies=frequencies, events=isi)
mor,phase = morlet(grid,dec=dec,events=blocks,frequencies=frequencies)
mor2, phase = morlet(grid,dec=dec,events=blocks,frequencies=frequencies,normFreqs=norms)

plot_morlet(mor2,grid,ws,frequencies,name='morletDriverNorm')
plot_morlet(mor,grid,ws,frequencies,name='morletDriver')
