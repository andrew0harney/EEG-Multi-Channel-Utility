import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SignalManager import SignalManager
from SignalManager import show_events_on_chan,photodiode_signal


#Script to check that events are aligned with some independent measurement
#This is useful for checking that log times match with the correct points in the signal
#Outputs a plot showing the independent signal and interesting event times



data_path = #Path to data
log_path =  #Path to log
#   
grid = SignalManager(data_path,log_file=log_path)
#
print 'Generating'
wanted_events = [2,3]
colours = ['r','b']

indpntSignal = photodiode_signal(grid)
#show_events_on_chan(grid,'C127',wanted_events,colours=event_colours) Would also highlight those events
print 'Plotting'
a = grid.data(columns=['DiodeChannelName'])
plt.plot(indpntSignal)
plt.hold(True)
plt.plot(a)
plt.plot()
plt.show()
