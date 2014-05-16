 Environment
---------------------

All code was written in Eclipse with the pydev plugin.
The Enthought Canopoy python distribution was used, although other environments such as anaconda could work equally as well. 
I also highly recommend adding your python package lib to your PYTHONPATH. Simply add "export PYTHONPATH=$PYTHONPATH:~/Enthought/Canopy_64bit/User/bin" to your .bashrc

Install pydev in Eclipse:
	http://pydev.org/manual_101_install.html

Using the Canopy (or whatever environment):
	Project > Properties > PyDev- Interpreter > Click here to configure ...
	an interpreter not listed > New > Interpreter Name = 'Canopy', Executable = '~/Enthought/Canopy_64/User/bin/python'

	Equally, the last step can is just an indication of your environment, so other package managers could be put here.

Required Packages:
	MNE: https://www.nmr.mgh.harvard.edu/martinos/userInfo/data/MNE_register/index.php
		Install to the system (note: mne_edf2fiff is the only executable required, so it may be easiest just to put this in /usr/bin when done)
	

	All other requirements can be found in the appropriate module docs
