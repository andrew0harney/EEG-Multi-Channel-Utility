import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SignalManager import SignalManager
from signalUtils import longest_event,mask_inter_block_signal
import tables

#The script demonstrates the initial preperation of the SignalManager and useful functionality at this stage.
#The assumption here is that the experimental log ran on a different clock from the photodiode. As such we have a log file and an offet file.
#The log contains the experimental times as per the computer clock
#The offset log contains the block times as per the signal clock

data_path = #
log_path =  #
offsets_path = #

#Create a new grid with the log file and offset the events by those describes in the offsets. Output a corrected log for future use.
grid = SignalManager(data_path,log_file=log_path,offsets=offsets_path,new_log_out = True)

#Create perfect signal against which we can test any analysis 
perfectSig = grid.photodiode_signal()
perfectSig = mask_inter_block_signal(grid,perfectSig)

#Create 4 blocks each with regular signals at 0.5hz, 2hz, 4hz, and 15hz. Note the number of blocks here should be commesurate with that of the data
freqBlocks = [0.5,2,4,15]
for i,(on,off) in enumerate(grid.blocks().values):
    b = perfectSig[grid.snap_time(on):grid.snap_time(off)]
    t = np.arange(len(b))*2*np.pi/float(grid.fs())
    wave = np.sin(t*freqBlocks[i])
    perfectSig[grid.snap_time(on):grid.snap_time(off)] = wave*b

#Add the perfect periodic waves to the data set. Also add one with some noise to simulate ideal data
grid.add_channel(perfectSig, 'PerfectSignal')
grid.add_channel(perfectSig+(np.random.randn(len(grid.times()))*0.5), 'PerfectSignalNoise')

#After this we check the validity of the channels and realise that some are bad (e.g using the channelValidity script), so let's remove them....
data_path = #
new_log_path = #

grid = SignalManager(data_path,log_file=new_log_path)
baseChannels = [chan for chan in grid.channels()]
baseChannels.append('C127')
grid.set_wd(channels=baseChannels)

badChans = ['badChan1Name','badChan2Name','badChanIName']
for chan in badChans:
    grid.remove_channel(chan)
