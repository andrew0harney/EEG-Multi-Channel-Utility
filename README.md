Multi-channel-management-utility
================================

Python implementation of a multiple channel signal management class. Has useful methods for loading, saving, retrieving, and splicing. Also helps with the management of epochs describing events in the data. 

Channel data is taken from from .edf, .fiff, and .hd5 formats. Log/event data is taken from .csv format. The framework essentially acts as a wrapper around the pandas toolkit. 

--------------------------------------
Requirements:

- numpy
- pandas
- mne (for format conversion) https://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php


----------------------------------------
TODO:

- More efficent reading and writing of data in pandas
- Selection of data from disk based on blocks


---------------------------------------

Notes:

- While some effort was made to keep this code general, it was ultimately written with a specific project in mind. However, I hope it may provide some use and it should still be straightforward to adapt. 

----------------------------------------

Please also refer to to the accompanying license before using any of this code.
