#############################################################################
#
# Example: Cross Validation 
#
# This example demonstrates, how n-fold cross-validation can be implemented
# in the Python Experiment Suite by using the 'repetitions' mechanism.
# 
# Each time the reset() function is called, it will reload the data and
# run another experiment, independent of all previous ones. No information
# is shared between repetitions. This makes it a bit tricky, if the dataset
# needs to be shuffled before (because all repetitions need the same 
# permutation of the dataset). Here, this is implemented by generating a
# unique hash-key from the dataset, whose first 7 digits are then used
# to initialize the random number generator. This guarantees the same
# split into batches for a given dataset. Use the flag shuffle=True if
# you want to randomize the dataset.
#
# Run this script from the command line on a single core: python suite.py -n1
#
# The output shows, how the dataset (here an array of consecutive
# numbers up to 10) is split into training and testing sets for each
# repetition. Change the repetitions number in the config file for
# other splits.
#
# In order to use cross-validation in your experiments, copy the method
# crossvalidation() to your suite class and pass the dataset array in from
# the reset() method, as shown here.
#
# For cross-validation experiments, the API retrieval function 
# get_histories_over_repetitions() is well suited to calculate the mean
# and variance over the repetitions. Section 6.8 in the documentation 
# explains its usage.
#
# Copyright (c) 2010 - Thomas Rueckstiess
#
#############################################################################

from expsuite import PyExperimentSuite
from numpy import *
import hashlib
import os

class MySuite(PyExperimentSuite):
    
    def __init__(self):
        PyExperimentSuite.__init__(self) 
        self.dataset = None
    
    def crossvalidation(self, dataset, params, rep, shuffle=True):
        """ This method takes a dataset in form of a numpy array of shape n x d,
            where n is the number of data points and d is the dimensionality of 
            the data. It further requires the current params dictionary and the
            current repetition number. The flag 'shuffle' determines, if the
            dataset should be shuffled before returning the training and testing
            batches. There will be params['repetitions'] many equally sized batches, 
            the rest of the dataset is discarded.
        """
        if params['repetitions'] < 2:
            raise SystemExit('%i-fold cross validation does not make sense. Use at least 2 repetitions.'%params['repetitions'])
        
        key = int(hashlib.sha1(dataset).hexdigest()[:7], 16)
        indices = range(dataset.shape[0])
        if shuffle:
            # create permutation unique to dataset
            random.seed(key)
            indices = random.permutation(indices)
               
        batchsize = dataset.shape[0] // (params['repetitions'])
        if batchsize == 0:
            raise SystemExit('Too many repetitions for cross-validation with this dataset. Max. number of repetitions is %i.'%dataset.shape[0])
        
        batch = []
        for i in range(params['repetitions']):
            batch.append(dataset[indices[batchsize*i:batchsize*(i+1)]])

        # create training and test data
        trainset = zeros((0, batch[0].shape[1]), dtype=dataset.dtype)
        for i,b in enumerate(batch):
            if i == rep:
                testset = b
            else:
                trainset = r_[trainset, b]
        
        return trainset, testset
        
        
    def reset(self, params, rep):
        """ To use the cross-validation mechanism, pass a numpy array of size n x d
            together with params and rep to the self.crossvalidation() method. 
            n is the number of samples and d is the dimensionality of the data. 
            The method returns a training and testing array.
        """
        data = load(os.path.join(params['dataset']))
        self.train, self.test = self.crossvalidation(data, params, rep, shuffle=True)
        
        # output for demonstration purposes
        print
        print
        print 'repetition %i:'%rep
        print 'training set', self.train.flatten()
        print 'testing set', self.test.flatten()
        print 'iterations (one per dot)', 
        
    def iterate(self, params, rep, n):
        """ In this example, the iterate method does nothing but print a dot and
            return the repetition and iteration number.
        """
        print '.',
        
        # use self.train and self.test here for your experiment...
        
        # return current repetition and iteration number and the 2 parameters
        ret = {'rep':rep, 'iter':n}
        return ret        

if __name__ == '__main__':
    mysuite = MySuite()
    mysuite.start()
    

