#############################################################################
#
# Example: Basic 
# 
# This example demonstrates the basic setup of the Python Experiment Suite.
# Create a class and inherit from PyExperimentSuite. Then implement the two
# functions reset() and iterate(). In reset() load any data required 
# throughout the whole experiment and initialize any variables and objects
# you might need. In iterate(), execute one calculation step and return
# any information in a dictionary, that you want to save to the log files.
#
# Run this script from the command line: python suite.py
#
# A directory called 'results' was created. It contains a subdirectory
# with the name of the experiment: myexperiment. In this directory, you
# can find the log files, one for each reptition. Each logfile contains
# one line for each iterate() call.
#
# Finally, have a look at the showresults.py script. Here, some simple
# API calls are demonstrated to retrieve your data.
#
# For more information on how these experiment types work, refer to the
# documentation.pdf, Section 3.
#
# Copyright (c) 2010 - Thomas Rueckstiess
#
#############################################################################

from expsuite import PyExperimentSuite

class MySuite(PyExperimentSuite):
    
    def reset(self, params, rep):
        """ for this basic example, nothing needs to be loaded or initialized."""
        pass
        
    def iterate(self, params, rep, n):
        """ this function does nothing but access the two parameters alpha and
            beta from the config file experiments.cfg and return them for the 
            log files, together with the current repetition and iteration number.
        """
        # access the two config file parameters alpha and beta
        alpha = params['alpha']
        beta = params['beta']
        
        # return current repetition and iteration number and the 2 parameters
        ret = {'rep':rep, 'iter':n, 'alpha':alpha, 'beta':beta}
        return ret        

if __name__ == '__main__':
    mysuite = MySuite()
    mysuite.start()

