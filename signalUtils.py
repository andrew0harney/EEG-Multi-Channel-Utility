import mne.time_frequency as mtf
import matplotlib.pyplot as plt
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__GridUtils__')

"""Utility functions for events and signals"""
__author__ = 'Andrew O\'Harney'

"""Normalisation Functions"""
def meanDesign(numPoints,events):
    """Calculates the mean on columns of the full events matrix
    Keyword Arguments:
    numPoints -- Number of encoding columns
    events -- Iterable returning design for each event"""
    logger.info('Calculating design mean')
    designMean = np.zeros(numPoints)
    N = 0
    for X,times in events:
        designMean += np.sum(X[:self.__longestEvent__,:],axis=0)
        N += len(times)
    return designMean / N

def l1Norm(numPoints,events):
    """Calculates the l1 normalisation parameter on columns of the design matrix
    Keyword Arguments:
    numPoints -- Number of encoding columns
    events -- Iterable returning design for each event"""
    logger.info('Calculating l1Norm')
    l1 = np.zeros(numPoints)
    for X,_ in events:
        l1 += np.sum(np.abs(X[:self.__longestEvent__,:]),axis=0)    
    return np.sqrt(l1)

def l2Norm(numPoints,events):
    """L2 norm for real valued event matrix
    Keyword Arguments:
    numPoints -- Number of encoding columns
    events -- Iterable returning design for each event"""
    
    logger.info('Calculating l2Norm')
    l2 = np.zeros(numPoints)
    for X,_ in events:
        l2 += np.sum(X[:self.__longestEvent__,:]**2,axis=0)    
    return np.sqrt(l2)
    
def varDesign(numPoints,events,mean):
    """Calculates variance on columns of design matrix
    Keyword Arguments:
    numPoints -- Number of encoding columns
    events -- Events to find mean signal of
    mean=None -- Mean of events"""
    logger.info('Calculating design variance')
    designVar = np.zeros(numPoints)
    N = 0
    for X,times in events:
        designVar += np.sum((X[:self.__longestEvent__]-mean)**2,axis=0)
        N += len(times)
    return designVar / (N-1)


"""Utility Functions"""
def photodiode_signal(grid,onEvents=None):
    """Generates a photodiode signal from the log file (useful for checking alignment with some independent signal)
    Keyword Arguments:
    onEvents -- Event codes to use as on periods for photodiode """
    
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
    """Returns a zero masked signal (inter-block set to 0)        
    Keyword Arguments:
    signal -- Signal to mask"""
        
    #Removes inter-block noise from the each channel
    mask = np.zeros(len(grid.times()))
    for on,off in grid.blocks().values:
        on = grid.time_to_index(on)
        off = grid.time_to_index(off)
        mask[on:off] = 1
    return mask*signal
 

def show_events_on_chan(grid,chan,eventCodes,colours=None):
    """Will highlight event points in a given channel
    Keyword Arguments:
    grid - Signal set to use use
    chan -- Channel to highlight points from
    eventCodes -- Events to use
    colours=None -- Colours to highlight respective event codes"""

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
    """Returns the longest event in events in number of points
    events - Events to find maximum of"""
    return events.apply(lambda x: grid.num_points(times=[x['pulse.on'],x['pulse.off']]) ,axis=1).max()

def shortest_event(grid,events):
    """Returns the shortest event in events
    events - Events to find shortest of"""
    return events.apply(lambda x: grid.num_points(times=[x['pulse.on'],x['pulse.off']]) ,axis=1).min()


def calculate_average(grid,events,norms=None,chans=None):
    """Calculates the average signal of the events
    Keyword Arguments:
    events -- Events to find me of
    norms=None -- Normalisationed events
    chans=None -- Channels to perform operation on"""
        
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
    """Returns normalised power spectrums for events (using morlet wavelets) [n_epochs,n_channels,n_frequencies]
    frequencies -- Morlet frequencies
    nc -- Number of cycles (DEFAULT 12*np.arange(1,len(frequencies)+1) #Linear formula [Chan,Baker et al. JNS 2011])
    Dec -- Decimation factor
    Events - events to find power of"""
    
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
    """Finds the up and down crossings in the signal
    sig - Signal to be thresholded
    thresh - The threshold for a crossing to occur
    channel - Channel in the signal to be used
    boost - Will polarise the signal (i.e Vx > 0 -> x = 1)"""
        
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

def smooth(x,window_len=11,window='hanning'):
    """Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Keyword Arguments:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    import numpy

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y
