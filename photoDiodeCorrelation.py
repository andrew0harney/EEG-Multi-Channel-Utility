import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Grid import Grid
from Grid import show_events_on_chan,photodiode_signal


#Script to check that events are aligned with some independent measurement
#This is useful for checking that log times match with the correct points in the signal
#Outputs a plot showing the independent signal and interesting event times



data_path = #Path to data
log_path =  #Path to log
#   
grid = NewBrain(data_path,log_file=log_path)
#
print 'Generating'
wanted_events = [2,3]

diodeSignal = photodiode_signal(grid)
#event_colours = ['g','b','y']
#show_events_on_chan(grid,'C127',wanted_events,colours=event_colours)
print 'Plotting'
a = grid.data(columns=['C127'])
plt.plot(diodeSignal[2000000:]*0.02)
plt.hold(True)
plt.plot(a[2000000:])
plt.plot()
plt.show()
