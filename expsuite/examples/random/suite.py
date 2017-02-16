#############################################################################
#
# Example: Random Numbers (random) 
# 
# This example demonstrates how to verify a a simple stochastic law,
# namely that the sample mean of random numbers drawn from a normal dis-
# tribution does converge to the actual mean of the distribution. We use
# the random module of the numpy package and store the drawn numbers in
# a numpy array.
# 
# Furthermore, this example makes use of the optional restore support.
# The restore_supported flag is set to True and the two functions
# save_state() and restore_state() are implemented. If the experiment
# is interrupted during a repetition, it will continue exactly from
# where it left off. (The default behavior was to restart the whole
# repetition and delete all iterations that were already executed).
#
# Run this script from the command line: python suite.py
#
# The results can be plotted with the matplotlib module. If it is
# available on your machine, run the plotresults.py script from the
# command line and look at the plots.
#
# For more information on the restore support functionality, refer to
# the documentation.pdf file, Section 3.4. The API interface is explained
# in more detail in Section 6.
#
# Copyright (c) 2010 - Thomas Rueckstiess
#
#############################################################################

from expsuite import PyExperimentSuite
from numpy import *
import os

class MySuite(PyExperimentSuite):
    
    restore_supported = True
    
    def reset(self, params, rep):
        # initialize array
        self.numbers = zeros(params['iterations'])
        
        # seed random number generator
        random.seed(params['seed'])
        
    def iterate(self, params, rep, n):
        # draw normally distributed random number
        self.numbers[n] = random.normal(params['mean'], params['std'])
        
        # calculate sample mean and offset
        samplemean = mean(self.numbers[:n+1])
        offset = abs(params['mean']-samplemean)
       
        # return dictionary
        ret = {'n':n, 'number':self.numbers[n], 
            'samplemean':samplemean, 'offset':offset}
        
        return ret
        
    def save_state(self, params, rep, n):
        # save array as binary file
        save(os.path.join(params['path'], params['name'], 
            'array_%i.npy'%rep), self.numbers)

    def restore_state(self, params, rep, n):
        # load array from file
        self.numbers = load(os.path.join(params['path'], 
            params['name'], 'array_%i.npy'%rep))
        

if __name__ == '__main__':
    mysuite = MySuite()
    mysuite.start()

