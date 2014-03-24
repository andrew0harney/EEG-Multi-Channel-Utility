import pandas as pd
import glob
import os
import numpy as np
import gc
import mne.time_frequency as mtf
import matplotlib.pyplot as plt
import logging
import mne

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__SignalManager__')

class SignalManager:
    #This class manages signals in edf,fif,or hd5 format (note all files are converted to hd5)
    #It makes extensive use of pandas to represent time series and events logs
    #requirements - pandas, numpy,matplotlib
    #             - mne (utility functions)
    #             - matplotlib (utility functions/plotting)
    
    
    ############Member Variables################
    __base_file_name = None
    __signals = None #Pandas Data frame columns=chans, index = times
    __log_file = None #Events log file
    __wd = None #Working data
    __wc = None #Working Channels
    __eventskeys = None #Codes for events in log
    __currentMeanCalcChans = None #Channels used to calculate current mean
    __currentMeanApplyChans = None #Channels means were applied to
    
    ###############Private Methods##########################
    
    
    def __init__(self,base_file_name=None,log_file=None,offsets=None,new_log_out=False,eventsKey=None):
        #Initialiser takes the path of the signal data
        #Can also set the event matrix or generate it from a log path
        #
        #Requires : base_file_name - the path and name of the signal data with no file extension
        #Optional : log_file - path to log file of events
        #         : offsets : if a log file has been provided, startimes is a path to a file containing appropriate offets for each block of events (if required - often useful for alignment)
        #         : new_log_out - if Offsets is specified then new_log_out is a Boolean value if the corrected file is to be output
        
        self.__base_file_name = base_file_name
        logger.debug('Using file : %s'%(base_file_name))
        self.__load_data__()      
        
        if eventsKey is None:
            eventsKey = {'_':0,'blockStart':1}
        self.set_eventsKey(eventsKey)
        
        #Check for log file to create event matrix
        if log_file:
            self.set_log_file(log_file,offsets,new_log_out)
        else:
            logger.info( 'No log specified -- assuming event matrix is in the data')
    

    def __load_data__(self):
        #Attempts to load data from .hd5
        #If .hd5 file does not exist, it will try to convert to it
        #
        try:
            if self.base_file_name() is None:
                raise Exception('Data was not specified')
            elif self.__check_for_files__('hd5'): #HD5 is the basis for pytables
                logger.info( 'Found .hd5 -- opening')
            elif self.__check_for_files__('fif'):
                logger.info( 'Could not find .hd5 -- converting  .fif->.hd5')
                self.__fif_2_hdf5__()
            elif self.__check_for_files__('edf'):
                logger.info( 'Could not find .hd5 -- converting .edf->.fif->.hd5')
                self.__edf_2_fif__()
                self.__fif_2_hdf5__()
            else:
                logger.info( "Could not find any appropriate files. Valid files are *.[edf, fif, hd5]. Assuming data will be supplied later")
            self.__open_hdf5__()
        except Exception as e: raise e

    def __check_for_files__(self, ftype):
        #Returns a list of files in the local Directory containing the base_file_name
        return glob.glob(self.__base_file_name + '.' + ftype)
    
    def __open_hdf5__(self):
        #Tries to open hd5 file
        try:
            self.__signals = pd.HDFStore(self.__base_file_name+'.hd5')
        except:
            logger.warning('Could not open hd5 file')
            raise Exception('Could not open hd5 file')

    def __edf_2_fif__(self):
        #Tries to convert edf to fif
        sysString = 'mne_edf2fiff --edf '+self.base_file_name()+'.edf --fif ' + self.base_file_name()+'.fif'
        logger.debug( sysString)
        try:
            os.system(sysString)
            logger.info( 'Conversion edf->fif complete')
        except:
            logger.warning('Could not find mne on system path')
            raise Exception('Could not find mne on system path -- cannot convert from .edf')
            
    def __fif_2_hdf5__(self):
        #Tries to convert .fif file to .hd5 format
        
        #Get data from the raw .fif file
        try:
            raw = mne.fiff.Raw(self.__base_file_name+'.fif')
        except:
            logger.warning('Could not open fif file')
            raise Exception("'Could not open fif file'")
        
        logger.debug( 'Extracting data from .fif')
        data,time_stamps = raw[1:,:raw.last_samp]
        ch_names = raw.ch_names[1:]
        logger.debug('Found channels : %s'%(str(ch_names)))
        fs = raw.info['sfreq']
        logger.debug('Found frequency : %f'%(fs))
        raw.close()
        self.save_hdf(data,time_stamps,ch_names,np.array(fs,self.base_file_name()))
        self.__open_hdf5__()

    def __create_events_matrix__(self):
        #Creates a Dataframe with index=data timestamps times, columns=signal channels
        logger.info( "Generating event matrix")
        events = pd.read_csv(self.__log_file,delimiter='\t')
        logger.debug( 'Found columns:'+str(events.columns))
        #self.__signals['event_matrix'] = events[['pulse.on','pulse.off','event.code']]
        self.__signals['event_matrix'] = events
        self.__flushSignals__()
        logger.info( "Saving event matrix")
        self.__find_blocks__()
          

    def __find_blocks__(self):
        #Finds the on and off times of blocks
        #Note: Blocks are defined as starting at event types blockStart(event id 1)
        
        logger.info( "Finding blocks")
        logger.debug( '\tCalculating block indices')
        em = self.event_matrix()
        blockStartIndices = em[em['event.code'] == self.__eventskey['blockStart']].index #Start of each block
        logger.debug(blockStartIndices)
    
        blockEndIndices = blockStartIndices
        blockEndIndices = blockEndIndices[1:].values - 2 #Remove the first pulse and shift to become last pulse in each preceding block
        blockEndIndices = np.append(blockEndIndices, len(em) - 1) #Add final pulse in file
        #Define the times of each block
        logger.debug( '\tCalculating start and end times of each block')
        
        startTimes = em.ix[blockStartIndices]['pulse.on'].values
        endTimes = em.ix[blockEndIndices]['pulse.off'].values
        logger.debug('Start times '+str(startTimes))
        logger.debug('End times '+str(endTimes))
        blocks = pd.DataFrame([startTimes,endTimes])
        blocks = blocks.T
        blocks.columns=['pulse.on','pulse.off']
        logger.info( "Saving blocks")
        self.__signals['blocks'] = blocks
        self.__flushSignals__()
    
    def __flushSignals__(self):
        #Forces a write to the hdf file
        self.__signals.flush()
           
    ###############Public Methods#################################

    def set_eventsKey(self,eventsKey):
        #Set a dictionary containing event code descriptions
        self.__eventskey = eventsKey
    
    def eventsKey(self):
        #Get events key
        return self.__eventskey    
    

    @staticmethod
    def save_hdf(data,times,cnames,fs,base_file_name):
        #Takes raw data and saves to HD5
        #data - raw signal data
        #cnames - channel names
        #base_file_name - base file name
        #fs - sample rate of data
        
        (x,y)= data.shape
        #Store in hd5(pytables) format
        logger.info( "Converting to pytables")
        signals = pd.HDFStore(base_file_name+'.hd5','w',complevel=9)
        #
        logger.debug( '\tSaving timing info')
        signals['times'] = pd.Series(times,dtype='float64')
        #
        logger.debug( '\tSaving data')
        signals['data']=pd.DataFrame(data.T,columns=cnames,index=times) #Ideally this would be tables=True
        #        
        logger.debug( "\tSaving meta data")
        signals['channels'] = pd.Series(cnames)
        signals['fs'] = pd.Series(fs)
        #signals['data_dimensions'] = pd.Series(['channels', 'samples'])
        signals.close()
        logger.info( 'Conversion complete')
    

    def add_channel(self,sig,name):
        #Adds a channel to the hd5 file
        #params sig - the raw signal to be added
        #       name - name of the new channel
        #       addToWd - add this channel to the working data
        #       mean - 'global' or 'local' (i.e use all channels or only those in the working data set)
        
        if name not in self.channels():
            newData = self.data()
            newData[name] = pd.Series(sig,name=[name],index=self.data().index)
            self.__signals['data']  = newData
            self.__signals['channels'] = self.channels().append(pd.Series(name,index=[len(self.channels())]))
            self.__signals.flush()
        else:
            logger.info( 'Channel with that name already exists')
             
    def remove_channel(self,chan):
        #Removes channel chan from the persistent .hd5 file
        
        #Try and remove the specified channel
        try:
            self.__signals['data'] = self.data().drop(chan,axis=1)
            #Remove from channel record
            self.__signals['channels'] = self.channels()[self.channels()!=chan]
            self.__flushSignals__()
        
            #If the channel was in the current working set
            currentChan = self.wc()
            if chan in currentChan:
                currentChan.remove(chan)
            
                if self.__currentMeanCalcChans is not None and chan in self.__currentMeanCalcChans:
                    self.set_wd(currentChan,meanCalcChans=[mc for mc in self.__currentMeanCalcChans if mc != chan],meanApplyChans=[mc for mc in self.__currentMeanApplyChans if mc != chan])
                else:   
                    self.set_wd(currentChan)
        except:
            logger.info( 'No channel called '+chan)
        
 
    #################Public Methods#########################   
    def blocks(self):
        #Return the on and off times of blocks
        return self.__signals['blocks']
    
    
    def base_file_name(self):
        #Return the base file path
        return self.__base_file_name
    
    
    def log_file_name(self):
        #Returns the log path of the psychopy file in use (i.e the file the event_matrix was generated from)
        return self.__log_file
    
    def event_matrix(self,types=None):
        #Returns the event matrix
        return self.__signals['event_matrix']

        
    def calc_mean(self,channels):
        #Return mean of data across all channels
        #channels - channels to calculate the mean over
    
        return pd.Series(self.data(columns=channels).mean(axis=1),index=self.times())
    
    
    def set_mean(self,meanCalcChans=None,meanApplyChans=None):
        #Remove a mean value from channels
        #meanCalcChans : channels to calculate the mean from - mean will be applied to these channels unless meanApplyChans is specified
        #meanApplyChans : channels to apply the mean to  (default is meanCalcChans)
        
        if meanCalcChans is not None:
            logger.info( 'Calculating mean')
            m = self.calc_mean(meanCalcChans)
            permMeanChans = []
            for chan in meanApplyChans if (meanApplyChans is not None) else meanCalcChans:
                self.__wd[chan] -=m #Cannot use .sub() - blows up!
                permMeanChans.append(chan)
            self.__currentMeanCalcChans = permMeanChans
            return m
        

    def data(self,columns=None):
        #Efficiently get data chunks from disk by supplying a column list (Note:data must be in table format)
        #Optional: columns - the channels to pull from the data
        if columns is not None:
            try:
                #Efficient read from disk (no need to load all data in memory) if data is in table format
                d= self.__signals.select('data', [pd.Term('columns','=',columns)])
            except:
                #If in pytables format then we need to load all data into memory and clip
                d= self.__signals['data'][columns]
        else:
            d= self.__signals['data']
        
        gc.collect() #Do garbage collection to free up wasted memory
        return d
    
    def times(self):
        #Returns the timestamps of samples
        return self.__signals['times']
    
    def channels(self):
        #Returns all channel names
        return self.__signals['channels']

    def wd(self,channels=None):
        #Return the current working data
        if self.__wd is not None:
            return self.__wd[self.wc() if channels is None else channels]
        else:
            logger.debug( "No working data was set")
       
    def fs(self):
        #Return the sample rate of the signal
        return self.__signals['fs'][0]
    

    def correct_event_times(self,offsets,new_log_out=False):
            #Correct the event matrix to include the appropriate block offsets
            #Required : Offsets - a Pandas Dataframe or Series with ['time'] offsets for each block
            #Optional : new_log_out - Boolean value if the corrected file is to be output
    
            logger.info( 'Correcting times in log file')
            offsets = pd.read_csv(offsets)
            blocks = self.blocks()
            startTimes = blocks['pulse.on']
            
            offsets = pd.Series(offsets['time']-startTimes,index=range(len(offsets))) #Remove the psychopy start time from the offset
            #Correct block times by the offsets
            logger.debug( "\tCorrecting blocks data")
            blocks['pulse.on']+=offsets
            blocks['pulse.off']+=offsets
            self.__signals['blocks'] = blocks
            logger.debug(blocks)
                
            logger.debug( '\tCorrecting event times')
            em = self.event_matrix()
            for i,block in enumerate(em['Block'].unique()):
                em.ix[em['Block']==block,'pulse.on']+= offsets.ix[i]
                em.ix[em['Block']==block,'pulse.off']+= offsets.ix[i]
            
            self.__signals['event_matrix'] = em
            self.__flushSignals__()
        
            if new_log_out:
                logger.info( "Saving corrected log file")
                self.__signals['event_matrix'].to_csv(self.__base_file_name+'_corrected_log.csv')
                self.__log_file = self.__base_file_name+'_corrected_log.csv'
            
    def set_log_file(self,log,offsets=None,new_log_out=False):
        #Sets the psychopy log file
        #Required: log - path to log file
        #Optional: offsets - path to file contains offsets of block times
        #        : new_log_out - Boolean value if the corrected file is to be output
        logger.info( 'Saving log file')
        self.__log_file = log
        self.__create_events_matrix__()
        if offsets:
            self.correct_event_times(offsets,new_log_out)

    def set_wd(self,channels=None,meanCalcChans=None,meanApplyChans=None):
        #Sets the working data to the selected channels (selects all channels by default)
        #Optional: columns - list of channels to use
        #          meanChans - The channels to calculate the mean from
        #          meanApplyChans - The channels to apply the mean to (default = meanCalcChans)
    
        logger.info( "Loading working data")
        self.__wd = self.data(columns=channels if channels else self.channels())
        self.__wc = channels if channels else self.channels()

        if meanCalcChans is not None:
            self.set_mean(meanCalcChans, meanApplyChans)
    
    def wc(self):
        #Returns a list of channel names for the working data
        return list(self.__wc)
    
    def set_fs(self,fs):
        #Set the frequency that the data was sampled at
        self.__signals['fs']=fs
        self.__flushSignals__() 

    def splice(self,data=None,times=None,indices=None):
        #Returns the signal specified between two time points
        #Optional: data - the data to splice (default is the whole data set)
        #          times - the start and end times to splice between
        #          indices - the start and end indices
        
        if data is None:
            data = self.wd()
        
        if times:
            return data.ix[self.snap_time(min(times)):self.snap_time(max(times))].values[:-1]
        elif indices:
            return data.iloc[indices]
    
    #Timing functions
    def snap_time(self,t):
        #Finds the nearest time index to time t
        return self.time_to_index(t)/float(self.fs())
        #return self.times()[self.time_to_index(t)]

    def index_to_time(self,ix):
       #Returns the time of a given index
       return self.times().iloc[ix]

    def time_to_index(self,t):
        #Returns the index of a given time point
        return int(np.floor(t*float(self.fs())))
        #return self.times().searchsorted(t)

    def num_points(self,times):
        #Returns the number of (inclusive) samples between two data points
        #If fs is specified then use that, otherwise will need to snip a section and check the length
        return self.time_to_index(max(times))-self.time_to_index(min(times))
    
    
#######Utility Functions##############
def photodiode_signal(grid):
#Generates a photodiode signal from the log file (useful for checking alignment with some independent signal)
    
    #Get all white pulse events
    em = grid.event_matrix()
    em = em[em['event.code'] != grid.eventsKey()['_']]
    diodeSignal = pd.Series(np.zeros(len(grid.times())),index=grid.times())
    
    #Set all diode pulses on
    em = em[em['Colour']=='white']
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
    #Returns the shortes event in events
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
    
    
     
def threshold_crossings(grid,sig=None,events=None,thresh=None,channel=None,tol=0):
    #Finds the up and down crossings in the signal
    #sig - signal to be thresholded
    #thresh - the threshold for a crossing to occur
    #channel - channel in the signal to be used
    #boost - Will polarise the signal (i.e Vx > 0 -> x = 1)
        
    if sig is None:
        if channel is not None:
            sig = grid.wd()[channel]
        else:
            sig = grid.wd()['C127']
        
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
