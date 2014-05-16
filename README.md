Multi-channel-management-utility
================================

The SignalManager class is the main class used for managing multi-channel signals and events in those signals. It has a number of useful functions for loading, splicing, controlling timing ect based on events, and for aligning the data with event epochs in the first place. See the docs folder for the API for this class. It also provides automatic conversion of file format from .edf or .fiff to .hd5. Most of the data is stored as Pandas Dataframes or Series so get comfortable with that. 

Signal data
------------

The class maintains the majority of information about the grid signal through the private Dataframe __signal. This directly interfaces with the .hd5 to provide efficient disk/memory trade-off. Be careful to note that this means if more than one SignalManger is instantiated at a given time, then it will be modify the same data. Inspection of the hd5 or code shows that it has the __signal Dataframe following fields:

  - times : A time vector for each sample in the data
  - data : this is the multi channel data
  - channels : A Series of channel names
  - fs : the sample frequency

At instantiation, all data will reside on disk. It will be up to you to manage what you want to load into memory. This is done by controlling the working data (__wd) and working channels (__wc) through the API.

At load time the SignalManager allows for the averaging across channels, as a form of detrending. It allows you to specify the channels that the average will be calculated from and the channels that it should be applied to. These respective sets are held in the private variables __currentMeanCalcChans and __currentMeanApplyChans. These will be managed appropriately with the removal of a signal from the data.

TODO:

  - More efficient reading and writing of data in pandas
  - Selection of data from disk based on blocks


Event Data
------------

In addition the other major data the class manages an experimental log, which is stored in the private Dataframe __log_file. This log is used to index and retrieve parts of the signal relevant to a given event. You can also work with individual events as Dataframes through the appropriate API functions. The log structure can be seen in the example (or the other private data) log csv file. This should be fairly flexible to store experimental information, but as a minimum each row represents an event in the experiment and should contain the following fields:

  - pulse.on : Start time of the event
  - pulse.off : End time of the event
  - event.code : Code for the class of event at that time
  - block : Which block (if any) this event belongs to. 

To begin with, the experimental times you have may not match due to differences in timing clocks ect. The SignalManager has an optional parameter (offsets) that specifies an offset for the times in each block (if you don’t plan on offsetting blocks in the experiment, then the block variable obviously isn’t important). It also has a parameter to output a new log file containing the corrected log times for future use.

This is a primitive way of shifting the timing of a number of events, and if instead the timing of up and down events from an independent channel is to be more precisely configured, then this method will not be suitable. For such a set up you will need to edit the times in the event log directly. 

A key part of the event storage is the management of event codes. In the example data (and private scripts) the independent photodiode signals the beginning of a block via a long on pulse followed by a long off. These are encoded as 0 and 1 respectively in the log file. Note that by default the SignalManager will look for 0 and 1 to signal the beginning of the block. If there are no blocks in your data then you can just put a false 0 row followed by a 1 row indicating the start of the experiment (this is done in the example csv provided). Also note that block start times given through the API start at the beginning of the long off.

The SignalManager also maintains a private dictionary called events key. This was intended to allow for the easier relation between event names a numbers. It has not been fully implemented as the simple convention of uniquely numbering each class of event has sufficed. This may be something you wish to revisit in future however. 

----------------------------------------------------------------------------------

TODO
------

  - Full implementation of {label:eventKey} mapping
  
Notes
------

- While some effort was made to keep this code general, it was ultimately written with a specific project in mind. However, I hope it may provide some use and it should still be straightforward to adapt. 
- Please also refer to to the accompanying license before using any of this code.
