import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scistats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import mne.time_frequency as mtf


#FFT that will zero pad a signal with NANs.
#This allows for greater resolution in the frequency domain
def nanFFT(x,n):
    fft = np.fft.fft(np.nan_to_num(x),n=n) #fft of non-nan components
    return fft

#Simple calculation for the power of an fft bin
def powerSpec(x,n=None):
    if n is None:
        n = len(x)
    x = nanFFT(x,n)
    return np.square(abs(x))

#Will save the short time fourier transform
def save_stft(powMap,ws,delta,grid,fs=None,channels=None,name=None):
    
    if fs is None:
        fs = grid.fs()
    if channels is None:
        channels = grid.wc()
    if name is None:
        name = 'STFT.pdf'
    elif name.find('.pdf') == -1:
        name += '.pdf'
    
    freq = np.fft.fftfreq(int(ws*fs),1.0/fs)
    freq = freq[:len(freq)/2]
    (x,y,z) = powMap.shape
    #pp = PdfPages(grid.wc()[0]+'-'+grid.wc()[-1]+'_'+name)
    fig = plt.figure()

    for cnum,chan in enumerate(channels):
        #
        plt.pcolor(powMap[cnum,:,:10].T,norm=LogNorm(vmin=powMap[cnum,:,:10].min(), vmax=powMap[cnum,:,:10].max()),cmap='blues')
        #
        xTickMultiple=200
        xlabels = [t for (i,t) in enumerate(np.arange(y)*delta/fs) if not i%xTickMultiple]
        plt.xticks(np.arange(len(xlabels)),xlabels,rotation=45)
        plt.axes().xaxis.set_major_locator(plt.MultipleLocator(xTickMultiple))
        #
        yTickMultiple = 1
        ylabels = [hz for (i,hz) in enumerate(freq[:11]) if not i%yTickMultiple]
        plt.yticks(np.arange(len(ylabels)),ylabels)
        plt.axes().yaxis.set_major_locator(plt.MultipleLocator(yTickMultiple))
        
        plt.ylabel('Hz')
        #
        plt.xlabel('Seconds')
        plt.title('STFT Power '+chan)
        plt.colorbar()
        #
        print 'Saving channel '+chan
        plt.savefig('STFT_'+chan)
        #pp.savefig()
        fig.clf()
        fig = plt.figure(name)
    
    #pp.close()


def morlet(grid,dec=None,frequencies=None,fs=None,events=None,nc=None,normFreqs=None):
    #Performs a morlet wavelet analysis across events

    if events is None:
        events = grid.event_matrix()
    if fs is None:
        fs = grid.fs()
    if dec is None:
        dec = 1
    if frequencies is None:
        frequencies= np.arange(1,101,dec)
    if nc is None:
        nc = 0.12*np.arange(1,len(frequencies)+1)
        print nc
        
    sigs = grid.wd()
    maxtimediff = (events['pulse.off']-events['pulse.on']).max()
    maxtimepoints =  round(maxtimediff*fs)+1
    data = np.zeros([len(events),len(grid.wc()),maxtimepoints])
    
    for k,chan in enumerate(grid.wc()):
        for j,(on,off) in enumerate(events.values):
            print 'Pre-processing Channel:'+chan + ' | Event '+str(j)+'/'+str(len(events.values))
            d=grid.splice(sigs[chan], times=[on,off])
            data[j,k,:len(d)] = d
    pows,phase=mtf.induced_power(data, Fs=float(fs), frequencies=frequencies, use_fft=False, n_cycles=nc, decim=dec, n_jobs=1,normFreqs=normFreqs)   
    return pows,phase


def mne_stft(grid,ws,delta,sigs=None,fs=None,events=None):
    #Performs a stft transform across the events????????????
    
    if events is None:
        events = grid.event_matrix()
    if fs is None:
        fs = grid.fs()
    if sigs is None:
        sigs = grid.wd()    
    
    ws*=fs
    delta*=fs
    maxtimediff = (events['pulse.off']-events['pulse.on']).max()
    maxtimepoints =  round(maxtimediff*fs/delta) - np.floor(ws/delta)
    powMap = np.zeros([len(events),len(grid.wc()),ws])
    data = np.zeros([len(events),len(grid.wc()),maxtimepoints])
    
    for j,(on,off) in enumerate(events.values):
        for k,chan in enumerate(grid.wc()):
            d=grid.splice(sigs[chan], times=[on,off])
            data[j,k,:len(d)] = mtf.stft(x=d,wsize=ws,tstep=delta)

    return powMap.mean(axis=2)
    
    

def plot_morlet(morlet,grid,ws,frequencies=None,fs=None,dec=None,name=None):
    
    if fs is None:
        fs = grid.fs()
    if dec is None:
        dec = 1
    if frequencies is None:
        frequencies = np.arange(1,101,1)
    if name is None:
        name = 'Morlet'
    
    (x,y,z) = morlet.shape
    times = np.arange(0,z/float(fs),ws)
    
    for k,chan in enumerate(grid.wc()):

        x = morlet[k,:,:]
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        data, freqs, bins, im = ax1.specgram(x)
        ax1.axis('tight')
        # We need to explictly set the linear threshold in this case...
        # Ideally you should calculate this from your bin size...
        ax2.set_yscale('symlog', linthreshy=0.01)
        ax2.pcolormesh(bins, freqs, 10 * np.log10(data))
        ax2.axis('tight')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Induced power (%s)' % chan)
        plt.savefig(name+chan)
        plt.clf()
        plt.figure()
    
def stft(grid,ws,delta,sigs=None,fs=None,events=None):
    #Performs a short time sliding window fourier transform
    #Averages a moving window across signals
    #Note that the algorithm processes an entire signal 
    #and moves to the next instead of loading all signals
    #into memory and averaging down the signals
    
    
    if events is None:
        events = grid.event_matrix()
    if fs is None:
        fs = grid.fs()
    if sigs is None:
        sigs = grid.wd()    
    
    ws*=fs
    delta*=fs
    maxtimediff = (events['pulse.off']-events['pulse.on']).max()
    maxtimepoints =   round(maxtimediff*fs/delta) - np.floor(ws/delta)

    powMap = np.zeros([len(grid.wc()),maxtimepoints,ws])

    for k,chan in enumerate(grid.wc()):
        #
        #Iterate events
        counter = np.zeros([maxtimepoints,ws])#Keep track of the number of events that contribute to the total time band
        for j,(on,off) in enumerate(events.values):
            print 'Processing Channel:'+chan + ' | Event '+str(j)+'/'+str(len(events.values))
            #
            #Move along each event, shifting the window by delta
            d=grid.splice(sigs[chan], times=[on,off])
            starts = np.linspace(0, int(len(d))-ws, maxtimepoints)#range(0,int(len(d)),int(delta))
            #
            for i,start in enumerate(starts):
                #
                #Get the current window splice
                s = d[start:start+ws]
                if not len(s)%2: #Ensure d is evenly sized for efficiency
                    s = s[:-1]
                #
                pwr = powerSpec(s)
                powMap[k,i,:len(pwr)] += pwr
                counter[i,:len(pwr)]+= np.ones(len(pwr))
        powMap[k,:,:]= (powMap[k,:,:]/counter) #Get the mean across the channel
    powMap[np.isnan(powMap)]=0 #Remove nan columns
    return powMap

def calcFFT(grid,events,ws=None):
    from andyClasses.NewBrain2 import shortest_event
    
    if ws is None:
        ws = shortest_event(grid,events)
        print '\tUsing minimum event size of '+str(ws)+' as size of FFT window'

    if ws%2:
        ws -= 1
        print '\tReducing FFT window size to be even (efficiency)'
    
    
    #FFT across all signals and events
    blockFFTs = np.zeros([len(grid.wc()),len(events),ws])
    
    for i,chan in enumerate(grid.wc()):
        sig  = grid.wd(channels=[chan])[chan]

        print "\tCalulcating fft for Channel: "+chan
        for row,(on,off) in enumerate(events[['pulse.on','pulse.off']].values):
            block = grid.splice(data=sig,times=[on,off])[:ws] #Slice out the event and truncate to the length of the shortest
            blockFFTs[i,row,:] = block
            blockFFTs[i,row,:] = np.power(np.abs(np.fft.fft(blockFFTs[i,row,:])),2)
    return blockFFTs

def save_fft(grid,stims,isis,name='',maxFreq=None):
        
    pp = PdfPages(name+'.pdf')
    (x,y,z) = stims.shape
    (x2,y2,z2) = isis.shape
    if x!=x2:
        print 'Number of channels does not match'
        print x,x2
        return
    elif z != z2:
        print 'FFTs are uneven lengths'
        print z,z2
        return
    elif y!= y2:
        print 'Warning - Number of stimuli events does not match the number of baseline periods'

    chans = grid.wc()
    freq = np.fft.fftfreq(z,1.0/grid.fs())
    freq = freq[:z/2]
    maxFreq = freq[-1] if maxFreq is None else maxFreq 
    freq = freq[np.where(freq<maxFreq)]
    z = len(freq)
    
    for i in range(x):
        print 'Saving channel '+chans[i]
        plt.figure()
        plt.hold(True)
        plt.title(chans[i])
               
        stimprc50 = np.percentile(stims[i,:,:z],50,axis=0)
        stimprc25 = np.percentile(stims[i,:,:z],25,axis=0)
        stimprc75 = np.percentile(stims[i,:,:z],75,axis=0)
        
        isiprc50 = np.percentile(isis[i,:,:z],50,axis=0)
        isiprc25 = np.percentile(isis[i,:,:z],25,axis=0)
        isiprc75 = np.percentile(isis[i,:,:z],75,axis=0)
            
        
        plt.loglog(freq,stimprc50,'b',label='Mean stimulus')
        plt.loglog(freq,stimprc25,'b.')
        plt.loglog(freq,stimprc75,'b--')
        
        plt.loglog(freq,isiprc50,'g',label='Mean ISI')
        plt.loglog(freq,isiprc25,'g.')
        plt.loglog(freq,isiprc75,'g--')

                      
        plt.legend(loc=1)
        plt.grid(True,which='both')
        pp.savefig()
        plt.clf()
        plt.close()
        
    pp.close()
