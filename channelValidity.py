import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Grid import Grid
from matplotlib.backends.backend_pdf import PdfPages

#This script produces histograms of channel powers for all interesting event periods
#It is useful as a pre-processing step to discard spurious channels
#Outputs a pdf of histograms for each channel

data_path = ##PATH TO DATA
log_path =  #PATH TO LOG

grid = NewBrain(data_path,log_file=log_path)
baseChannels = [chan for chan in grid.channels()]
grid.set_wd(channels=baseChannels)

plt.plot(grid.wd(channels=['C127'])['C127'])
blocks = grid.blocks()
em=grid.event_matrix()

events = [2,3]
pp = PdfPages('ChannelHists_events.pdf')

for channel in grid.wc():
    print 'Processing '+channel
    plt.figure()
    x = grid.wd(channels=[channel])[channel] #Get full channel
    for i,(event) in enumerate(events):
        e = em[em['event.code']==event] #Get the interesting event
        y = np.array([])
        #
        for on,off in e[['pulse.on','pulse.off']].values:#Get portion of siganl relating to event
            y = np.concatenate((y, x[grid.time_to_index(on):grid.time_to_index(off)+1]))

        plt.subplot(2,1,i)
        plt.title('Event '+str(event))
        
        #Check channel for validity with histogram
        hist, bins = np.histogram(y, bins=50)
        width = (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        #
        plt.bar(center, hist, align='center', width=width)
        plt.xlabel('A(s)')
        plt.ylabel('#')
        plt.suptitle(channel)
    plt.tight_layout()
    pp.savefig()
    plt.clf()
    plt.close()
        
pp.close()
