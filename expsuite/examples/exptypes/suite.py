#############################################################################
#
# Example: Experiment Types (exptypes) 
# 
# This example demonstrates the 3 different experiment types that the
# Experiment Suite currently supports: grid, list and single.
# 
# First, have a look at the experiments.cfg file. Notice the 3 experiments
# defined: gridexp, listexp, singleexp. Each of them inherits the same
# parameters from the DEFAULT section, in particular alpha and beta, but
# defines a different type of experiment.
#
# Now run the example from the command line: python suite.py
# 
# Then go into the results directory and look at the different experiments.
# Notice how the grid experiment has 9 sub-experiments, one for each
# combination of parameters. The list experiment on the other hand only 
# contains 3 experiments, one for each parameter pair with the same index. 
# Finally, the single experiment type does not have any sub-experiments, 
# but the values assigned to the alpha and beta are the original lists from 
# the config file.
# 
# For more information on how these experiment types work, refer to the
# documentation.pdf, Section 4.3 and Figure 2.
#
# Copyright (c) 2010 - Thomas Rueckstiess
#
#############################################################################


from expsuite import PyExperimentSuite

class MySuite(PyExperimentSuite):
    
    def reset(self, params, rep):
        """ for this simple example, nothing needs to be loaded or initialized."""
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

