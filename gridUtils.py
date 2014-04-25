import mne.time_frequency as mtf
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__GridUtils__')

#######Utility Functions##############
def photodiode_signal(grid,channel=['C127'],onEvents=None):
#Generates a photodiode signal from the log file (useful for checking alignment with some independent signal)
    
    if onEvents is None:
        logger.info('Must specify the on events of the photodiode')
        return
    
    #Get all white pulse events
    em = grid.event_matrix()
    diodeSignal = pd.Series(np.zeros(len(grid.times())),index=grid.times())
    
    #Set all diode pulses on
    em = em[em['event.code'].isin(onEvents)]
    for (on,off) in em[['pulse.on','pulse.off']].values:
        diodeSignal[grid.snap_time(on):grid.snap_time(off)] = 1
    return pd.Series(diodeSignal,index=grid.times())
    


def mask_inter_block_signal(grid,signal=None):
    #Returns a zero masked signal (inter-block set to 0)        
    #signal - signal to mask
        
    #Removes inter-block noise from the each channel
    mask = np.zeros(len(grid.times()))
    for on,off in grid.blocks().values:
        on = grid.time_to_index(on)
        off = grid.time_to_index(off)
        mask[on:off] = 1
    return mask*signal
 

def show_events_on_chan(grid,chan,eventCodes,colours=None):
#Will highlight event points in a given channel
#Grid - Signal set to use use
#chan - channel to highlight points from
#eventCodes - events to use
#colours - colours to highlight respective event codes



    #Plot the base signal
    signal = grid.data()[chan]
    plt.plot(signal)    
    plt.hold(True)

    if colours is None:
        colours = ['r','b','g','y','p']

    #For each event type highlight the appropriate region in the signal
    em = grid.event_matrix()
    for i,event in enumerate(eventCodes):
        highlight = em[em['event.code']==2] #UPDATE : take function handle for acceptability criteria 
        blockOnIx = highlight[['pulse.on','pulse.off']].apply(lambda x: [grid.time_to_index(x['pulse.on']), grid.time_to_index(x["pulse.off"])],axis=1)
        blockOnIx.apply(lambda x: plt.axvspan(x['pulse.on'], x["pulse.off"], facecolor=colours[i%len(colours)], alpha=0.5),axis=1)
    plt.title('Psychopy/EEG line-up')
    plt.xlabel('Time (s)')
    plt.ylabel('EEG')


def longest_event(grid,events):
    #Returns the longest event in events in number of points
    #events - events
    return events.apply(lambda x: grid.num_points(times=[x['pulse.on'],x['pulse.off']]) ,axis=1).max()

def shortest_event(grid,events):
    #Returns the shortest event in events
    #events - events
    return events.apply(lambda x: grid.num_points(times=[x['pulse.on'],x['pulse.off']]) ,axis=1).min()

#######Utility Functions##############

def calculate_average(grid,events,norms=None,chans=None):
        
        
    if chans is None:
        chans = grid.wc()
    if norms is not None:
        if len(norms) != len(events):
            logger.info( 'Number of events and baselines is not equal')
            return
        norms.columns = ['baseline.on','baseline.off']
               
    maxtimediff = (events['pulse.off']-events['pulse.on']).max()
    maxtimepoints =  round(maxtimediff*grid.fs())+1
    data = grid.wd()
        
    avrg = np.zeros([len(chans),maxtimepoints])
        
    for j,chan in enumerate(chans):
        logger.info( 'Processing channel '+chan)
        counter = np.zeros(maxtimepoints)
        for i,(on,off) in enumerate(events[['pulse.on','pulse.off']].values):
            sig = grid.splice(data[chan], times=[on,off])
            bl = norms.iloc[i].values
            bl = grid.splice(data[chan],times=[bl[0],bl[1]])
            sig -= bl[:len(sig)].mean()               
            avrg[j,:len(sig)] += sig
            counter[:len(sig)] += np.ones(len(sig))
            avrg[j,:]/=counter
    return avrg
    
    
def normSignal(grid,frequencies = None,nc=None,dec=None,events=None):
    #Returns normalised power spectrums for events (using morlet wavelets) [n_epochs,n_channels,n_frequencies]
    #frequencies - morlet frequencies
    #nc - number of cycles
    #dec - decimation
    #events - events to find power of
    
    if frequencies is None:
        frequencies = np.arange(1,101)
    if nc is None:
        nc = 0.12*np.arange(1,len(frequencies)+1) #Linear formula [Chan,Baker et al. JNS 2011]
        print nc
    if dec is None:
        dec = 1
    if events is None:
        events=grid.event_matrix()
        events = events[events['event.code']==5] #Use ISIs as baseline
     
    #Pre allocate power matrix for efficiency
    pows = np.zeros([len(events),len(grid.wc()),len(frequencies)])
        
    for j,chan in enumerate(grid.wc()):
        for i,(on,off) in enumerate(events[['pulse.on','pulse.off']].values):
            logger.info( 'Normalising power on channel '+chan+' baseline event '+str(i+1)+'/'+str(len(events)))
            data = grid.splice(grid.wd()[chan],times=[on,off])[None,None,:]
            data = np.vstack([data,data]) #mne doesn't allow only 1 event so duplicate the same event (which will then be averaged)
            p,phase=mtf.induced_power(data, Fs=grid.fs(), frequencies=frequencies, use_fft=False, n_cycles=nc, decim=dec, n_jobs=1,normFreqs=None)
            pows[i,j,:]=p.mean(axis=2).squeeze() #Take average across time for each frequency
    return pows
    
    
     
def threshold_crossings(grid,sig=None,events=None,thresh=None,channel='C127',tol=0):
    #Finds the up and down crossings in the signal
    #sig - signal to be thresholded
    #thresh - the threshold for a crossing to occur
    #channel - channel in the signal to be used
    #boost - Will polarise the signal (i.e Vx > 0 -> x = 1)
        
    if sig is None:
        if channel is not None:
            sig = grid.wd()[channel]
        else:
            sig = grid.wd()[channel]
        
    if thresh is None:
        thresh = sig.mean()
        print 'Using '+str(thresh)+' as threshold'
        
    if events is None:
        events = grid.event_matrix()                  
        
    up_crosses = np.array([])
    down_crosses = np.array([])

    #plt.figure()
    #plt.hold(True)
    for (on,off) in events.values:
        signal = grid.splice(sig, times=[on-tol[0],off+tol[1]])
        #plt.plot(signal)
        ##all points above threshold 
        above = np.where(signal>=thresh)[0]
        ##if point one step back is below threshold, its an upcrossing: make sure not to include negative indices
        indices = above-1  
        up_cross = above[np.where(signal[indices] < thresh)[0]]+grid.time_to_index(on)
        
        ##if point one step ahead is below threshold, its a down-crossing: make sure not to go out of bounds
        indices = np.asarray(above)+1
        L = len(signal)
        indices = indices[indices < L]
        down_cross = above[np.where(signal[indices] < thresh)[0]]+grid.time_to_index(on)
        up_crosses = np.hstack((up_crosses,up_cross))
        down_crosses = np.hstack((down_crosses,down_cross))
            
    print 'Found '+str(len(up_crosses))+' up crossings and '+str(len(down_crosses))+' down crossings'
    up_crosses.sort()
    down_crosses.sort()
    return up_crosses, down_crosses