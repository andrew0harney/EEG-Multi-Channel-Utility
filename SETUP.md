 Environment 
------------------------
(To those who pick up the project...whoever you may be!)

All code was written in Eclipse with the pydev plugin.
The Enthought Canopoy python package manager was used, although other environments such as anaconda could work equally as well.


If you intend to use a package manager or otherwise some virtualenv I recommend adding the python lib to your PYTHONPATH if you intend to go on using your default python enviroment. 

For Canopy, simply add "export PYTHONPATH=$PYTHONPATH:~/Enthought/Canopy_64bit/User/bin" to your .bashrc


Install pydev in Eclipse:
	http://pydev.org/manual_101_install.html

Using the Canopy (or whatever package manager):
   Project > Properties > PyDev- Interpreter > Click here to configure
   an interpreter not listed > New > Interpreter Name = 'Canopy', Executable = '~/Enthought/Canopy_64/User/bin/python'

Required Packages:
   MNE: https://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php
   Install to the system (note: mne_edf2fiff is the only executable required, so it may be easiest just to put this in /usr/bin when done)
   This is only required for format conversion.

All other dependencies can be found in the appropriate module docs. In particular note that the python distribution of MNE is not the same as the one you installed to your system.

Also take care with this step if you are using a package manager or virtual enviroment to ensure that the libraries end up the the correct site-packages for python. 
